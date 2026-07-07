"""PermutedMNIST continual learning — the benchmark where trivial strategies fail.

Full from-scratch MLP (no frozen backbone): T tasks, each a fixed random pixel
permutation of MNIST (task 1 = identity), single shared 10-way head (domain-
incremental), replay-free. This is the classic VCL benchmark; per Dimitrije,
train-from-scratch + per-parameter posterior merging fails here (weight
permutation symmetry -> no alignment), so we include exactly that strategy
('scratch-merge') as a baseline alongside the VCL recursion ('ivon-cl').

Methods: adamw | adamw-ewc | ivon | ivon-cl | evon-cl | scratch-merge.
Supports the TD-VCL '-Hard' regime via --samples-per-task / --epochs.
"""
import argparse
import gzip
import json
import time

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
from jax import nn, vmap
import optax
from optax import tree_utils as otu

from blrax import ivon, evon, noisy_value_and_grad, get_scale, sample_posterior
from blrax.states import ScaleByIvonState, ScaleByEvonState, _is_evon_leaf

WD_EPS = 1e-7


# ---------------------------------------------------------------- data
def load_mnist(data_dir):
    def imgs(fn):
        with gzip.open(fn, 'rb') as f:
            return np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 784)

    def lbls(fn):
        with gzip.open(fn, 'rb') as f:
            return np.frombuffer(f.read(), np.uint8, offset=8)

    tr = imgs(f'{data_dir}/mnist_train_images.gz').astype(np.float32) / 255.0
    te = imgs(f'{data_dir}/mnist_test_images.gz').astype(np.float32) / 255.0
    m, s = tr.mean(), tr.std()
    return ((tr - m) / s, lbls(f'{data_dir}/mnist_train_labels.gz').astype(np.int32),
            (te - m) / s, lbls(f'{data_dir}/mnist_test_labels.gz').astype(np.int32))


def make_perms(n_tasks, seed, identity_first=False):
    rng = np.random.default_rng(seed)
    perms = ([np.arange(784)] if identity_first else [rng.permutation(784)])
    perms += [rng.permutation(784) for _ in range(n_tasks - 1)]
    return perms


# ---------------------------------------------------------------- model
def init_mlp(key, hidden):
    ks = jr.split(key, 3)
    def lin(k, i, o):
        return {'w': jr.normal(k, (i, o)) * np.sqrt(2.0 / i), 'b': jnp.zeros(o)}
    return {'l1': lin(ks[0], 784, hidden), 'l2': lin(ks[1], hidden, hidden),
            'l3': lin(ks[2], hidden, 10)}


def mlp_apply(p, x):
    h = nn.relu(x @ p['l1']['w'] + p['l1']['b'])
    h = nn.relu(h @ p['l2']['w'] + p['l2']['b'])
    return h @ p['l3']['w'] + p['l3']['b']


def tree_sum(t):
    return sum(jnp.sum(l) for l in jtu.tree_leaves(t))


# ---------------------------------------------------------------- training
def train_one_task(params, opt_state, optim, loss_fn, X, y, key, *, epochs, batch, bayes):
    @jax.jit
    def step(params, opt_state, x, yb, k):
        if bayes:
            l, g, opt_state = noisy_value_and_grad(loss_fn, opt_state, params, k, x, yb)
        else:
            l, g = jax.value_and_grad(loss_fn)(params, x, yb, k)
        u, opt_state = optim.update(g, opt_state, params)
        return optax.apply_updates(params, u), opt_state, jnp.mean(l)

    rng = np.random.default_rng(0)
    N = len(X)
    for _ in range(epochs):
        perm = rng.permutation(N)
        for s_ in range(max(1, N // batch)):
            b = perm[s_ * batch:(s_ + 1) * batch]
            key, k = jr.split(key)
            params, opt_state, _ = step(params, opt_state, X[b], y[b], k)
    return params, opt_state, key


def eval_tasks(params, Xte, yte, perms, upto):
    accs = []
    for j in range(upto + 1):
        lg = mlp_apply(params, Xte[:, perms[j]])
        accs.append(float((lg.argmax(-1) == yte).mean()))
    return accs


def fisher_diag_mlp(params, X, y, n_sub=2000):
    idx = np.random.default_rng(0).permutation(len(X))[:n_sub]
    def per_sample(p, x, yi):
        return optax.softmax_cross_entropy_with_integer_labels(
            mlp_apply(p, x[None]), yi[None]).mean()
    g = vmap(jax.grad(per_sample), in_axes=(None, 0, 0))(params, X[idx], y[idx])
    return jtu.tree_map(lambda a: (a ** 2).mean(0), g)


def run_method(method, hp, Xtr, ytr, Xte, yte, perms, key, *,
               epochs, batch, samples_per_task, seed):
    n_tasks = len(perms)
    key, hk = jr.split(key)
    params = init_mlp(hk, hp['hidden'])
    theta_init = params
    bayes = method.startswith(('ivon', 'evon', 'scratch'))
    cl = method.endswith('-cl')

    def mk_optim(zeta, lr):
        if method.startswith('adamw'):
            return optax.adamw(lr, weight_decay=hp.get('wd', 0.0))
        wd = WD_EPS if (cl or method == 'scratch-merge') else hp.get('wd', 1e-4)
        if method.startswith('evon'):
            return evon(lr, ess=zeta, hess_init=hp['h0'], weight_decay=wd,
                        clip_radius=hp.get('clip', float('inf')), b1=0.9, b2=hp['b2'],
                        b3=0.95, precond_every=10, max_precond_dim=10000)
        return ivon(lr, ess=zeta, hess_init=hp['h0'], weight_decay=wd,
                    clip_radius=hp.get('clip', float('inf')), b1=0.9, b2=hp['b2'])

    # per-task train subsets (the -Hard regime subsamples)
    rngs = np.random.default_rng(10_000 + seed)
    task_idx = [rngs.permutation(len(Xtr))[:samples_per_task] for _ in range(n_tasks)]

    mu_prev = lam_prev = None
    if cl:
        mu_prev = theta_init
        lam_prev = jtu.tree_map(lambda p: hp['p0'] * jnp.ones_like(p), params)
    fisher_cum = jtu.tree_map(jnp.zeros_like, params)
    ewc_anchor = params
    merged_lam = merged_nat = None                      # scratch-merge accumulators

    opt_state = None
    R, avg_curve = [], []
    for t in range(n_tasks):
        idx = task_idx[t]
        Xt, yt = Xtr[idx][:, perms[t]], ytr[idx]
        zeta = hp.get('ess_abs') or hp.get('tau', 1.0) * len(idx)
        lr_t = hp['lr'] if t == 0 else hp.get('lr2', hp['lr'])

        if method == 'scratch-merge':
            key, hk2 = jr.split(key)
            params = init_mlp(hk2, hp['hidden'])        # fresh net every task

        if cl:
            def loss_fn(p, x, yb, k, mu=mu_prev, lam=lam_prev, z=zeta):
                ce = optax.softmax_cross_entropy_with_integer_labels(
                    mlp_apply(p, x), yb).mean()
                quad = 0.5 * tree_sum(jtu.tree_map(
                    lambda l, pp, m: jnp.sum(l * (pp - m) ** 2), lam, p, mu))
                return ce + quad / z
        elif method == 'adamw-ewc' and t > 0:
            lam_e = jtu.tree_map(lambda f: hp['lam_ewc'] * f, fisher_cum)
            def loss_fn(p, x, yb, k, mu=ewc_anchor, lam=lam_e):
                ce = optax.softmax_cross_entropy_with_integer_labels(
                    mlp_apply(p, x), yb).mean()
                return ce + 0.5 * tree_sum(jtu.tree_map(
                    lambda l, pp, m: jnp.sum(l * (pp - m) ** 2), lam, p, mu))
        else:
            def loss_fn(p, x, yb, k):
                return optax.softmax_cross_entropy_with_integer_labels(
                    mlp_apply(p, x), yb).mean()

        optim = mk_optim(zeta, lr_t)
        if opt_state is None or not (cl and bayes) or method == 'scratch-merge':
            opt_state = optim.init(params)
        else:
            s = opt_state[0]
            if isinstance(s, ScaleByIvonState):
                s = s._replace(ess=zeta, weight_decay=WD_EPS,
                               momentum=otu.tree_zeros_like(s.momentum),
                               count=jnp.zeros([], jnp.int32))
            else:
                leaves = jtu.tree_map(
                    lambda l: l._replace(G_bar=jnp.zeros_like(l.G_bar)),
                    s.leaves, is_leaf=_is_evon_leaf)
                s = s._replace(ess=zeta, weight_decay=WD_EPS, leaves=leaves)
            opt_state = (s,) + opt_state[1:]

        key, tk = jr.split(key)
        params, opt_state, key = train_one_task(
            params, opt_state, optim, loss_fn, Xt, yt, tk,
            epochs=epochs, batch=batch, bayes=bayes)

        if cl:
            mu_prev = params
            lam_prev = jtu.tree_map(lambda s_: 1.0 / s_ ** 2, get_scale(opt_state[0]))
        if method == 'adamw-ewc':
            fisher_cum = jtu.tree_map(jnp.add, fisher_cum,
                                      fisher_diag_mlp(params, Xt, yt))
            ewc_anchor = params
        if method == 'scratch-merge':
            lam_t = jtu.tree_map(lambda s_: 1.0 / s_ ** 2, get_scale(opt_state[0]))
            nat_t = jtu.tree_map(lambda l, m: l * m, lam_t, params)
            if merged_lam is None:
                merged_lam, merged_nat = lam_t, nat_t
            else:
                merged_lam = jtu.tree_map(jnp.add, merged_lam, lam_t)
                merged_nat = jtu.tree_map(jnp.add, merged_nat, nat_t)
            eval_params = jtu.tree_map(lambda n, l: n / l, merged_nat, merged_lam)
        else:
            eval_params = params

        row = eval_tasks(eval_params, Xte, yte, perms, t)
        R.append(row)
        avg_curve.append(float(np.mean(row)))

    T = n_tasks
    metrics = {'ACC': float(np.mean(R[-1])),
               'AIA': float(np.mean(avg_curve)),
               'forgetting': float(np.mean([max(R[tt][j] for tt in range(j, T)) - R[-1][j]
                                            for j in range(T - 1)])),
               'LA': float(np.mean([R[tt][tt] for tt in range(T)])),
               'avg_curve': avg_curve}
    return {'method': method, 'hp': hp, 'R': R, 'metrics': metrics}


DEFAULT_HP = {
    'adamw':         {'lr': 1e-3, 'wd': 0.0, 'hidden': 256},
    'adamw-ewc':     {'lr': 1e-3, 'wd': 0.0, 'lam_ewc': 100.0, 'hidden': 256},
    'ivon':          {'lr': 0.05, 'h0': 0.5, 'b2': 0.99999, 'wd': 1e-4, 'tau': 10.0,
                      'clip': 0.01, 'hidden': 256},
    'ivon-cl':       {'lr': 0.05, 'h0': 0.5, 'b2': 0.99999, 'p0': 1.0, 'tau': 10.0,
                      'clip': 0.01, 'hidden': 256},
    'evon-cl':       {'lr': 0.05, 'h0': 0.5, 'b2': 0.99999, 'p0': 1.0, 'tau': 10.0,
                      'clip': 0.01, 'hidden': 256},
    'scratch-merge': {'lr': 0.05, 'h0': 0.5, 'b2': 0.99999, 'tau': 10.0,
                      'clip': 0.01, 'hidden': 256},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', default='/ptmp/cheins/blrax/examples/data')
    ap.add_argument('--methods',
                    default='adamw,adamw-ewc,ivon,ivon-cl,evon-cl,scratch-merge')
    ap.add_argument('--n-tasks', type=int, default=10)
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--samples-per-task', type=int, default=60000)
    ap.add_argument('--batch', type=int, default=128)
    ap.add_argument('--hidden', type=int, default=100)
    ap.add_argument('--identity-first', action='store_true')
    ap.add_argument('--seeds', default='0')
    ap.add_argument('--hp', default='{}')
    ap.add_argument('--out', default='results_cl_pmnist.json')
    args = ap.parse_args()

    print('jax', jax.__version__, '| devices', jax.devices(), flush=True)
    Xtr, ytr, Xte, yte = load_mnist(args.data_dir)
    Xtr, ytr = jnp.asarray(Xtr), jnp.asarray(ytr)
    Xte, yte = jnp.asarray(Xte), jnp.asarray(yte)
    print(f'mnist: train {Xtr.shape} test {Xte.shape} | spt={args.samples_per_task} '
          f'epochs={args.epochs}', flush=True)

    overrides = json.loads(args.hp)
    runs = []
    for seed in [int(s) for s in args.seeds.split(',')]:
        perms = make_perms(args.n_tasks, seed, identity_first=args.identity_first)
        for method in args.methods.split(','):
            hp = dict(DEFAULT_HP[method]); hp['hidden'] = args.hidden
            hp.update(overrides.get(method, {}))
            t0 = time.time()
            r = run_method(method, hp, Xtr, ytr, Xte, yte, perms,
                           jr.PRNGKey(1000 + seed), epochs=args.epochs,
                           batch=args.batch, samples_per_task=args.samples_per_task,
                           seed=seed)
            r['seed'] = seed; r['wall_s'] = round(time.time() - t0, 1)
            m = r['metrics']
            print(f"seed{seed} {method:14s} | ACC {m['ACC']*100:5.2f}%  "
                  f"AIA {m['AIA']*100:5.2f}%  forget {m['forgetting']*100:5.2f}  "
                  f"LA {m['LA']*100:5.2f}  ({r['wall_s']}s)", flush=True)
            runs.append(r)

    with open(args.out, 'w') as f:
        json.dump({'meta': vars(args), 'runs': runs}, f, indent=2, default=str)
    print('wrote', args.out, flush=True)


if __name__ == '__main__':
    main()
