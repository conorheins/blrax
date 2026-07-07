"""Continual learning on Split CIFAR-100 (10 tasks x 10 classes, class-incremental)
with a linear head on frozen DINOv2 features — Bayesian finetuning demo.

Methods (all replay-free, same single 100-way head, same seen-class masking):
  adamw      naive sequential finetuning (the catastrophic-forgetting baseline)
  adamw-ewc  AdamW + diagonal-Fisher EWC (literature anchor)
  ivon       naive IVON (fresh state per task, no prior recursion)
  ivon-cl    IVON + VCL recursion: carry opt_state AND penalize toward previous
             posterior mean with weight Lam_prev = 1/get_scale(state)^2
             (= zeta_t*(h_t+wd)), penalty coeff Lam_prev/(2*zeta_{t+1}) in the
             per-sample mean loss => zeta*penalty = 0.5||theta-mu_prev||^2_Lam.
  evon-cl    EVON v1: carry state + diagonal Kronecker-marginal penalty from
             get_scale (never reset count; zero G_bar at boundaries).
  joint      upper bound: one task with all 100 classes (adamw + ivon variants).

Protocol per the pretrained-backbone CL literature: single shared head, train
each task with CE masked to classes SEEN SO FAR, evaluate on the union of seen
classes; metrics ACC (final avg over task subsets), AIA, forgetting (Chaudhry),
BWT (Lopez-Paz), plus NLL/ECE (@mean and @BMA-32 for Bayesian methods).
"""
import argparse
import json
import time

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
from jax import nn, vmap
import optax
from optax import tree_utils as otu

from blrax import ivon, evon, noisy_value_and_grad, get_scale, sample_posterior
from blrax.states import ScaleByIvonState, ScaleByEvonState, _is_evon_leaf

WD_EPS = 1e-7
NEG = -1e9


# ---------------------------------------------------------------- data / tasks
def load_features(path, synthetic=False, n_classes=100, dim=384, seed=0):
    if synthetic:
        rng = np.random.default_rng(seed)
        centers = rng.normal(size=(n_classes, dim)).astype(np.float32) * 0.6
        ytr = np.repeat(np.arange(n_classes), 100)
        yte = np.repeat(np.arange(n_classes), 20)
        xtr = centers[ytr] + rng.normal(size=(len(ytr), dim)).astype(np.float32)
        xte = centers[yte] + rng.normal(size=(len(yte), dim)).astype(np.float32)
        return (jnp.asarray(xtr), jnp.asarray(ytr, jnp.int32),
                jnp.asarray(xte), jnp.asarray(yte, jnp.int32))
    d = np.load(path)
    return (jnp.asarray(d['train_x']), jnp.asarray(d['train_y'], jnp.int32),
            jnp.asarray(d['test_x']), jnp.asarray(d['test_y'], jnp.int32))


def make_tasks(train_y, test_y, n_tasks, seed):
    """Seeded class order -> per-task index arrays (host-side numpy)."""
    n_classes = int(train_y.max()) + 1
    order = np.random.default_rng(seed).permutation(n_classes)
    per = n_classes // n_tasks
    tasks = []
    ytr = np.asarray(train_y); yte = np.asarray(test_y)
    for t in range(n_tasks):
        cls = order[t * per:(t + 1) * per]
        tr_idx = np.where(np.isin(ytr, cls))[0]
        te_idx = np.where(np.isin(yte, cls))[0]
        tasks.append({'classes': cls, 'train_idx': tr_idx, 'test_idx': te_idx})
    return tasks, order


def make_tasks_dil(train_dom, test_dom, n_classes, seed):
    """Domain-incremental: task t = domain t (same label space every task).

    Domain order is seeded; 'classes' is the full label set so the seen-class
    mask is all-True from task 1 — forgetting here is pure domain interference,
    no logit suppression."""
    doms = np.unique(np.asarray(train_dom))
    order = np.random.default_rng(seed).permutation(doms)
    all_cls = np.arange(n_classes)
    tr_d = np.asarray(train_dom); te_d = np.asarray(test_dom)
    return [{'classes': all_cls,
             'train_idx': np.where(tr_d == d)[0],
             'test_idx': np.where(te_d == d)[0]} for d in order], order


# ---------------------------------------------------------------- head / loss
def init_head(key, dim, n_classes):
    return {'w': 0.01 * jr.normal(key, (dim, n_classes)), 'b': jnp.zeros(n_classes)}


def logits_fn(params, x, seen_mask):
    return x @ params['w'] + params['b'] + jnp.where(seen_mask, 0.0, NEG)


def tree_sum(t):
    return sum(jnp.sum(l) for l in jax.tree_util.tree_leaves(t))


def make_loss(seen_mask, mu_prev=None, lam_prev=None, zeta=None):
    """Per-sample mean CE (masked to seen classes) + optional quadratic prior."""
    def loss_fn(params, x, y, key):
        lg = logits_fn(params, x, seen_mask)
        ce = optax.softmax_cross_entropy_with_integer_labels(lg, y).mean()
        if mu_prev is not None:
            quad = 0.5 * tree_sum(jax.tree.map(
                lambda l, p, m: jnp.sum(l * (p - m) ** 2), lam_prev, params, mu_prev))
            ce = ce + quad / zeta
        return ce
    return loss_fn


# ---------------------------------------------------------------- evaluation
def evaluate(params, feats, labels, tasks, upto, num_bins=15):
    """R-row: accuracy on each task<=upto (masked to seen classes) + union metrics."""
    seen = np.concatenate([tasks[j]['classes'] for j in range(upto + 1)])
    seen_mask = jnp.zeros(params['b'].shape[0], bool).at[jnp.asarray(seen)].set(True)
    row = []
    for j in range(upto + 1):
        idx = tasks[j]['test_idx']
        lg = logits_fn(params, feats[idx], seen_mask)
        row.append(float((lg.argmax(-1) == labels[idx]).mean()))
    uni = np.concatenate([tasks[j]['test_idx'] for j in range(upto + 1)])
    lg = logits_fn(params, feats[uni], seen_mask)
    y = labels[uni]
    p = nn.softmax(lg, -1)
    acc_u = float((lg.argmax(-1) == y).mean())
    nll_u = float(optax.softmax_cross_entropy_with_integer_labels(lg, y).mean())
    conf = p.max(-1); pred = lg.argmax(-1)
    bins = jnp.linspace(0, 1, num_bins + 1); ece = 0.0
    for i in range(num_bins):
        m = (conf > bins[i]) & (conf <= bins[i + 1]); c = m.sum()
        ece += jnp.where(c > 0, jnp.abs((pred == y)[m].sum() / jnp.maximum(c, 1)
                                        - conf[m].sum() / jnp.maximum(c, 1)) * c, 0.0)
    return row, {'acc': acc_u, 'nll': nll_u, 'ece': float(ece / y.shape[0])}


def evaluate_bma(key, params, opt_state, feats, labels, tasks, upto, num_mc=32):
    seen = np.concatenate([tasks[j]['classes'] for j in range(upto + 1)])
    seen_mask = jnp.zeros(params['b'].shape[0], bool).at[jnp.asarray(seen)].set(True)
    state0 = opt_state[0]
    out = sample_posterior(key, params, state0, shape=(num_mc,))
    samples = out if isinstance(state0, ScaleByEvonState) else out[0]
    uni = np.concatenate([tasks[j]['test_idx'] for j in range(upto + 1)])
    x, y = feats[uni], labels[uni]
    probs = jnp.zeros((x.shape[0], params['b'].shape[0]))
    for i in range(num_mc):
        si = jax.tree.map(lambda a: a[i], samples)
        probs = probs + nn.softmax(logits_fn(si, x, seen_mask), -1)
    probs = probs / num_mc
    acc = float((probs.argmax(-1) == y).mean())
    nll = float(-jnp.log(probs[jnp.arange(y.shape[0]), y] + 1e-12).mean())
    return {'acc': acc, 'nll': nll, 'num_mc': num_mc}


# ---------------------------------------------------------------- EWC fisher
def fisher_diag(params, x, y, seen_mask):
    """Exact empirical-Fisher diagonal for the linear-softmax head."""
    p = nn.softmax(logits_fn(params, x, seen_mask), -1)
    err2 = (p - nn.one_hot(y, p.shape[-1])) ** 2          # (N, C)
    return {'w': jnp.einsum('nd,nc->dc', x ** 2, err2) / x.shape[0],
            'b': err2.mean(0)}


# ---------------------------------------------------------------- training
def train_task(params, opt_state, optim, loss_fn, xtr, ytr, key, *,
               epochs, batch, bayes):
    N = xtr.shape[0]
    steps = max(1, N // batch)

    @jax.jit
    def step(params, opt_state, x, y, k):
        if bayes:
            l, g, opt_state = noisy_value_and_grad(loss_fn, opt_state, params, k, x, y)
        else:
            l, g = jax.value_and_grad(loss_fn)(params, x, y, k)
        u, opt_state = optim.update(g, opt_state, params)
        return optax.apply_updates(params, u), opt_state, jnp.mean(l)

    rng = np.random.default_rng(0)
    for _ in range(epochs):
        perm = rng.permutation(N)
        for s in range(steps):
            b = perm[s * batch:(s + 1) * batch]
            key, k = jr.split(key)
            params, opt_state, _ = step(params, opt_state, xtr[b], ytr[b], k)
    return params, opt_state, key


def run_method(method, hp, feats_tr, ytr, feats_te, yte, tasks, key, *,
               epochs, batch, num_mc):
    dim = feats_tr.shape[1]
    n_classes = int(ytr.max()) + 1
    key, hk = jr.split(key)
    params = init_head(hk, dim, n_classes)
    theta_init = params

    bayes = method.startswith(('ivon', 'evon'))
    cl = method.endswith('-cl')
    ewc = method == 'adamw-ewc'

    def mk_optim(zeta):
        if method.startswith('adamw'):
            return optax.adamw(hp['lr'], weight_decay=hp.get('wd', 0.0))
        wd = WD_EPS if cl else hp.get('wd', 1e-4)
        if method.startswith('ivon'):
            return ivon(hp['lr'], ess=zeta, hess_init=hp['h0'], weight_decay=wd,
                        clip_radius=hp.get('clip', float('inf')), b1=0.9, b2=hp['b2'])
        return evon(hp['lr'], ess=zeta, hess_init=hp['h0'], weight_decay=wd,
                    clip_radius=hp.get('clip', float('inf')), b1=0.9, b2=hp['b2'],
                    b3=0.95, precond_every=10, max_precond_dim=10000)

    mu_prev, lam_prev = None, None
    if cl:                                   # base prior toward init (task 1)
        mu_prev = theta_init
        lam_prev = jax.tree.map(lambda p: hp['p0'] * jnp.ones_like(p), params)
    fisher_cum = jax.tree.map(jnp.zeros_like, params)   # EWC
    ewc_anchor = params

    opt_state = None
    R = []; union = []; bma = []
    seen_np = []
    for t, task in enumerate(tasks):
        tr = task['train_idx']
        zeta = hp.get('tau', 1.0) * len(tr)
        seen_np = np.concatenate([seen_np, task['classes']]).astype(int) if len(seen_np) else task['classes']
        seen_mask = jnp.zeros(n_classes, bool).at[jnp.asarray(seen_np)].set(True)

        # ---- loss for this task ----
        if cl:
            loss_fn = make_loss(seen_mask, mu_prev, lam_prev, zeta)
        elif ewc:
            loss_fn = make_loss(seen_mask, ewc_anchor,
                                jax.tree.map(lambda f: hp['lam_ewc'] * f, fisher_cum),
                                1.0) if t > 0 else make_loss(seen_mask)
        else:
            loss_fn = make_loss(seen_mask)

        # ---- optimizer / state at boundary ----
        optim = mk_optim(zeta)
        if opt_state is None or not (cl and bayes):
            opt_state = optim.init(params)          # fresh state (naive/adamw/task1)
        else:                                        # CL: NamedTuple surgery, never re-init
            s = opt_state[0]
            if isinstance(s, ScaleByIvonState):
                s = s._replace(ess=zeta, weight_decay=WD_EPS,
                               momentum=otu.tree_zeros_like(s.momentum),
                               count=jnp.zeros([], jnp.int32))
            else:                                    # EVON: keep count (basis bookkeeping)
                leaves = jax.tree.map(
                    lambda l: l._replace(G_bar=jnp.zeros_like(l.G_bar)),
                    s.leaves, is_leaf=_is_evon_leaf)
                s = s._replace(ess=zeta, weight_decay=WD_EPS, leaves=leaves)
            opt_state = (s,) + opt_state[1:]

        key, tk = jr.split(key)
        params, opt_state, key = train_task(params, opt_state, optim, loss_fn,
                                            feats_tr[tr], ytr[tr], tk,
                                            epochs=epochs, batch=batch, bayes=bayes)

        # ---- posterior -> prior snapshot (BEFORE next boundary surgery) ----
        if cl:
            mu_prev = params
            lam_prev = jax.tree.map(lambda s_: 1.0 / s_ ** 2, get_scale(opt_state[0]))
        if ewc:
            f = fisher_diag(params, feats_tr[tr], ytr[tr], seen_mask)
            fisher_cum = jax.tree.map(jnp.add, fisher_cum, f)
            ewc_anchor = params

        # ---- evaluation ----
        row, uni = evaluate(params, feats_te, yte, tasks, t)
        R.append(row); union.append(uni)
        if bayes:
            key, bk = jr.split(key)
            bma.append(evaluate_bma(bk, params, opt_state, feats_te, yte, tasks, t,
                                    num_mc=num_mc))

    T = len(tasks)
    acc_final = float(np.mean(R[-1]))
    aia = float(np.mean([np.mean(R[t]) for t in range(T)]))
    forg = float(np.mean([max(R[t][j] for t in range(j, T)) - R[-1][j]
                          for j in range(T - 1)]))
    bwt = float(np.mean([R[-1][j] - R[j][j] for j in range(T - 1)]))
    la = float(np.mean([R[t][t] for t in range(T)]))
    return {'method': method, 'hp': hp, 'R': R, 'union': union, 'bma': bma,
            'metrics': {'ACC': acc_final, 'AIA': aia, 'forgetting': forg,
                        'BWT': bwt, 'LA': la,
                        'union_final': union[-1], 'bma_final': bma[-1] if bma else None}}


# ---------------------------------------------------------------- main
DEFAULT_HP = {
    'adamw':     {'lr': 1e-3, 'wd': 0.0},
    'adamw-ewc': {'lr': 1e-3, 'wd': 0.0, 'lam_ewc': 100.0},
    'ivon':      {'lr': 1e-2, 'h0': 0.5, 'b2': 0.99999, 'wd': 1e-4, 'tau': 1.0},
    'ivon-cl':   {'lr': 1e-2, 'h0': 0.5, 'b2': 0.99999, 'p0': 1.0, 'tau': 1.0},
    'evon-cl':   {'lr': 1e-2, 'h0': 0.5, 'b2': 0.99999, 'p0': 1.0, 'tau': 1.0},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--features', default='/ptmp/cheins/data/cifar100_dinov2s14.npz')
    ap.add_argument('--synthetic', action='store_true')
    ap.add_argument('--scenario', choices=['cil', 'dil'], default='cil',
                    help='dil: tasks = domains from train_dom/test_dom in the npz')
    ap.add_argument('--methods', default='adamw,adamw-ewc,ivon,ivon-cl,evon-cl')
    ap.add_argument('--n-tasks', type=int, default=10)
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--batch', type=int, default=128)
    ap.add_argument('--num-mc', type=int, default=32)
    ap.add_argument('--seeds', default='0')
    ap.add_argument('--hp', default='{}', help='JSON dict method->hp overrides')
    ap.add_argument('--out', default='results_cl_split_cifar.json')
    args = ap.parse_args()

    print('jax', jax.__version__, '| devices', jax.devices(), flush=True)
    feats_tr, ytr, feats_te, yte = load_features(args.features, args.synthetic)
    mu = feats_tr.mean(0); sd = feats_tr.std(0) + 1e-6      # standardize features
    feats_tr = (feats_tr - mu) / sd; feats_te = (feats_te - mu) / sd
    print(f'features: train {feats_tr.shape} test {feats_te.shape}', flush=True)

    doms = None
    if args.scenario == 'dil':
        dd = np.load(args.features)
        doms = (dd['train_dom'], dd['test_dom'])

    overrides = json.loads(args.hp)
    all_runs = []
    for seed in [int(s) for s in args.seeds.split(',')]:
        if args.scenario == 'dil':
            tasks, order = make_tasks_dil(doms[0], doms[1], int(ytr.max()) + 1, seed)
        else:
            tasks, order = make_tasks(ytr, yte, args.n_tasks, seed)
        for method in args.methods.split(','):
            hp = dict(DEFAULT_HP[method]); hp.update(overrides.get(method, {}))
            t0 = time.time()
            r = run_method(method, hp, feats_tr, ytr, feats_te, yte, tasks,
                           jr.PRNGKey(1000 + seed), epochs=args.epochs,
                           batch=args.batch, num_mc=args.num_mc)
            r['seed'] = seed; r['wall_s'] = round(time.time() - t0, 1)
            m = r['metrics']
            line = (f"seed{seed} {method:10s} | ACC {m['ACC']*100:5.2f}%  "
                    f"AIA {m['AIA']*100:5.2f}%  forget {m['forgetting']*100:5.2f}  "
                    f"BWT {m['BWT']*100:+5.2f}  LA {m['LA']*100:5.2f}  "
                    f"| union acc {m['union_final']['acc']*100:5.2f}% "
                    f"nll {m['union_final']['nll']:.3f} ece {m['union_final']['ece']:.3f}")
            if m['bma_final']:
                line += (f" | BMA acc {m['bma_final']['acc']*100:5.2f}% "
                         f"nll {m['bma_final']['nll']:.3f}")
            line += f" ({r['wall_s']}s)"
            print(line, flush=True)
            all_runs.append(r)

    with open(args.out, 'w') as f:
        json.dump({'meta': vars(args), 'runs': all_runs}, f, indent=2, default=str)
    print('wrote', args.out, flush=True)


if __name__ == '__main__':
    main()
