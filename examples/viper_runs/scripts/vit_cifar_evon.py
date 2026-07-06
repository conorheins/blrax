"""EVON vs IVON vs h-EVON vs AdamW: a compact ViT on CIFAR-10/100.

A harder, more ill-conditioned task than the MNIST MLP — attention + MLP weight
matrices give structured curvature, which is where EVON's eigenbasis posterior
is meant to separate from IVON's diagonal one. Reports point-estimate (mean) and
Bayesian-model-averaging (BMA) acc / NLL / ECE. EVON runs exercise the eigenbasis
refresh / power-iteration path fixed by commit 91be154.

Runs on Viper-GPU (AMD MI300A, ROCm jax 0.8.2). blrax on PYTHONPATH; optax/
equinox/chex from the ~/blrax-rocm venv.
"""
import argparse
import json
import os
import pickle
import time
import traceback

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
from jax import lax, nn, vmap
import equinox as eqx
import optax

from blrax import ivon, evon, sample_posterior
from blrax.states import ScaleByEvonState
from training import compute_ece


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def _unpickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f, encoding='latin1')


def load_cifar(root, which='cifar100'):
    """Load the prepared {cifar10,cifar100}.npz (uint8 HWC images), standardise."""
    d = np.load(os.path.join(root, f'{which}.npz'))
    train_x = d['train_x'].astype(np.float32) / 255.0
    test_x = d['test_x'].astype(np.float32) / 255.0
    mean = train_x.reshape(-1, 3).mean(0)
    std = train_x.reshape(-1, 3).std(0)
    train_x = (train_x - mean) / std
    test_x = (test_x - mean) / std
    n_classes = int(d['n_classes'])
    return (jnp.asarray(train_x), jnp.asarray(d['train_y'], jnp.int32),
            jnp.asarray(test_x), jnp.asarray(d['test_y'], jnp.int32), n_classes)


# --------------------------------------------------------------------------- #
# compact ViT
# --------------------------------------------------------------------------- #
class Block(eqx.Module):
    n1: eqx.nn.LayerNorm
    n2: eqx.nn.LayerNorm
    qkv: eqx.nn.Linear
    proj: eqx.nn.Linear
    fc1: eqx.nn.Linear
    fc2: eqx.nn.Linear
    heads: int = eqx.field(static=True)

    def __init__(self, key, dim, heads, mlp_ratio=4):
        k = jr.split(key, 4)
        self.n1 = eqx.nn.LayerNorm(dim)
        self.n2 = eqx.nn.LayerNorm(dim)
        self.qkv = eqx.nn.Linear(dim, 3 * dim, key=k[0])
        self.proj = eqx.nn.Linear(dim, dim, key=k[1])
        self.fc1 = eqx.nn.Linear(dim, mlp_ratio * dim, key=k[2])
        self.fc2 = eqx.nn.Linear(mlp_ratio * dim, dim, key=k[3])
        self.heads = heads

    def __call__(self, x):                       # x: (seq, dim)
        seq, dim = x.shape
        hd = dim // self.heads
        h = vmap(self.n1)(x)
        qkv = vmap(self.qkv)(h).reshape(seq, 3, self.heads, hd).transpose(1, 2, 0, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]         # (heads, seq, hd)
        attn = nn.softmax((q @ k.transpose(0, 2, 1)) / jnp.sqrt(hd), axis=-1)
        o = (attn @ v).transpose(1, 0, 2).reshape(seq, dim)
        x = x + vmap(self.proj)(o)
        h = vmap(self.n2)(x)
        return x + vmap(self.fc2)(nn.gelu(vmap(self.fc1)(h)))


class ViT(eqx.Module):
    patch: eqx.nn.Linear
    cls: jax.Array
    pos: jax.Array
    blocks: list
    norm: eqx.nn.LayerNorm
    head: eqx.nn.Linear
    psize: int = eqx.field(static=True)

    def __init__(self, key, n_classes, dim=128, depth=4, heads=4, patch=4, img=32, ch=3):
        ks = jr.split(key, depth + 4)
        npatch = (img // patch) ** 2
        self.patch = eqx.nn.Linear(patch * patch * ch, dim, key=ks[0])
        self.cls = jr.normal(ks[1], (1, dim)) * 0.02
        self.pos = jr.normal(ks[2], (npatch + 1, dim)) * 0.02
        self.blocks = [Block(ks[3 + i], dim, heads) for i in range(depth)]
        self.norm = eqx.nn.LayerNorm(dim)
        self.head = eqx.nn.Linear(dim, n_classes, key=ks[3 + depth])
        self.psize = patch

    def __call__(self, x):                       # x: (H, W, C)
        H, W, C = x.shape
        p = self.psize
        patches = (x.reshape(H // p, p, W // p, p, C)
                    .transpose(0, 2, 1, 3, 4)
                    .reshape((H // p) * (W // p), p * p * C))
        tok = vmap(self.patch)(patches)
        tok = jnp.concatenate([self.cls, tok], 0) + self.pos
        for b in self.blocks:
            tok = b(tok)
        return self.head(self.norm(tok[0]))


def n_params(model):
    p, _ = eqx.partition(model, eqx.is_array)
    return int(sum(x.size for x in jax.tree_util.tree_leaves(p)))


# --------------------------------------------------------------------------- #
# training / eval
# --------------------------------------------------------------------------- #
def run_training(key, model, optim, train_x, train_y, test_x, test_y, *,
                 epochs, batch, estimator):
    """Small compiled units only — the ROCm XLA compiler segfaults on the big
    fused graphs (full-dataset in-jit gather + 390-step scan + 10k eval). So:
    shuffle on the host, Python loop over steps with a tiny jitted single-step,
    and chunked jitted eval. Each compiled graph is one batch / one chunk."""
    import numpy as _np
    from blrax import noisy_value_and_grad
    params, static = eqx.partition(model, eqx.is_array)
    opt_state = optim.init(params)
    N = int(train_x.shape[0])
    steps = N // batch

    def loss_fn(params, x, y, *args):
        m = eqx.combine(params, static)
        return optax.softmax_cross_entropy_with_integer_labels(vmap(m)(x), y).mean()

    @jax.jit
    def train_step(params, opt_state, x, y, key):
        l, g, s = noisy_value_and_grad(loss_fn, opt_state, params, key, x, y, estimator=estimator)
        u, s = optim.update(g, s, params)
        # IVON's 'sampling' estimator returns a per-MC-sample loss (ndim 1); EVON
        # returns a scalar. Reduce to a scalar so downstream float() is safe.
        return optax.apply_updates(params, u), s, jnp.mean(l)

    @jax.jit
    def eval_chunk(params, x):
        return vmap(eqx.combine(params, static))(x)

    def evaluate(params):
        logits = jnp.concatenate(
            [eval_chunk(params, test_x[i:i + 2000]) for i in range(0, test_x.shape[0], 2000)], 0)
        acc = (logits.argmax(-1) == test_y).mean()
        nll = optax.softmax_cross_entropy_with_integer_labels(logits, test_y).mean()
        return float(acc), float(nll), float(compute_ece(logits, test_y))

    rng = _np.random.default_rng(0)
    curve = {'acc': [], 'nll': [], 'ece': [], 'loss': []}
    for _ in range(epochs):
        perm = rng.permutation(N)
        ep_loss = jnp.zeros(())
        for s in range(steps):
            bidx = perm[s * batch:(s + 1) * batch]
            key, sk = jr.split(key)
            params, opt_state, l = train_step(params, opt_state, train_x[bidx], train_y[bidx], sk)
            ep_loss = ep_loss + l
        acc, nll, ece = evaluate(params)
        curve['loss'].append(float(ep_loss) / steps)
        curve['acc'].append(acc)
        curve['nll'].append(nll)
        curve['ece'].append(ece)
    return eqx.combine(params, static), opt_state, {k: jnp.asarray(v) for k, v in curve.items()}


def plugin_eval(model, test_x, test_y):
    logits = vmap(model)(test_x)
    acc = (logits.argmax(-1) == test_y).mean()
    nll = optax.softmax_cross_entropy_with_integer_labels(logits, test_y).mean()
    return float(acc), float(nll), float(compute_ece(logits, test_y))


def bma_eval(key, model, opt_state, test_x, test_y, num_mc, n_classes):
    params, static = eqx.partition(model, eqx.is_array)
    state0 = opt_state[0]
    out = sample_posterior(key, params, state0, shape=(num_mc,))
    samples = out if isinstance(state0, ScaleByEvonState) else out[0]

    @eqx.filter_jit
    def softmax_pred(sp, x):
        return nn.softmax(vmap(eqx.combine(sp, static))(x), -1)

    probs = jnp.zeros((test_x.shape[0], n_classes))
    for i in range(num_mc):
        si = jax.tree.map(lambda a: a[i], samples)
        probs = probs + softmax_pred(si, test_x)
    probs = probs / num_mc
    logp = jnp.log(probs + 1e-12)
    idx = jnp.arange(test_y.shape[0])
    return {'acc': float((probs.argmax(-1) == test_y).mean()),
            'nll': float(-logp[idx, test_y].mean()),
            'ece': float(compute_ece(logp, test_y)), 'num_mc': num_mc}


def run_one(name, make_optim, estimator, key, model0, data, *, epochs, batch, num_mc, bayes):
    train_x, train_y, test_x, test_y, n_classes = data
    k_train, k_bma = jr.split(key)
    t0 = time.time()
    model, opt_state, metrics = run_training(
        k_train, model0, make_optim(), train_x, train_y, test_x, test_y,
        epochs=epochs, batch=batch, estimator=estimator)
    jax.block_until_ready(metrics)
    wall = time.time() - t0
    acc, nll, ece = plugin_eval(model, test_x, test_y)
    res = {'name': name, 'estimator': estimator, 'wall_clock_s': round(wall, 1),
           'n_params': n_params(model0), 'epochs': epochs,
           'plugin': {'acc': acc, 'nll': nll, 'ece': ece},
           'curve': {k: [float(v) for v in metrics[k]] for k in ('acc', 'nll', 'ece', 'loss')}}
    if bayes:
        res['bma'] = bma_eval(k_bma, model, opt_state, test_x, test_y, num_mc, n_classes)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='cifar100', choices=['cifar10', 'cifar100'])
    ap.add_argument('--data-root', default='/ptmp/cheins/data')
    ap.add_argument('--out', default='results_vit_cifar.json')
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--batch', type=int, default=128)
    ap.add_argument('--dim', type=int, default=128)
    ap.add_argument('--depth', type=int, default=4)
    ap.add_argument('--heads', type=int, default=4)
    ap.add_argument('--patch', type=int, default=4)
    ap.add_argument('--lr', type=float, default=1e-2)        # IVON / EVON peak lr
    ap.add_argument('--adamw-lr', type=float, default=1e-3)
    ap.add_argument('--hess-init', type=float, default=0.35)
    ap.add_argument('--wd', type=float, default=1e-4)
    ap.add_argument('--clip-radius', type=float, default=1e-1)
    ap.add_argument('--num-mc', type=int, default=32)
    args = ap.parse_args()

    print('jax', jax.__version__, '| devices', jax.devices(), flush=True)
    data = load_cifar(args.data_root, args.dataset)
    train_x, train_y, test_x, test_y, n_classes = data
    N = int(train_x.shape[0])
    steps = (N // args.batch) * args.epochs
    warmup = max(1, steps // 20)
    print(f'{args.dataset}: train={N} test={test_x.shape[0]} classes={n_classes} '
          f'| total_steps={steps} warmup={warmup}', flush=True)

    def sched(peak):
        return optax.warmup_cosine_decay_schedule(0.0, peak, warmup, steps, end_value=peak * 0.1)

    model0 = ViT(jr.PRNGKey(0), n_classes, dim=args.dim, depth=args.depth,
                 heads=args.heads, patch=args.patch)
    print(f'ViT dim={args.dim} depth={args.depth} heads={args.heads} patch={args.patch} '
          f'| params={n_params(model0):,}', flush=True)
    keys = dict(zip(('adamw', 'ivon', 'evon', 'h-evon'), jr.split(jr.PRNGKey(7), 4)))

    def mk_evon():
        return evon(sched(args.lr), ess=N, hess_init=args.hess_init, weight_decay=args.wd,
                    clip_radius=args.clip_radius, precond_every=10, max_precond_dim=10000,
                    one_sided=False, b2=0.9999)

    specs = [
        ('adamw', lambda: optax.adamw(sched(args.adamw_lr), weight_decay=args.wd),
         'sampling', keys['adamw'], False),
        ('ivon', lambda: ivon(sched(args.lr), ess=N, hess_init=args.hess_init,
                              weight_decay=args.wd, clip_radius=args.clip_radius),
         'sampling', keys['ivon'], True),
        ('evon', mk_evon, 'sampling', keys['evon'], True),
        ('h-evon', mk_evon, 'hutchinson', keys['h-evon'], True),
    ]

    runs = []
    for name, mk, est, key, bayes in specs:
        print(f'--- running {name} ({est}) ---', flush=True)
        try:
            r = run_one(name, mk, est, key, model0, data,
                        epochs=args.epochs, batch=args.batch, num_mc=args.num_mc, bayes=bayes)
            print(f"    done: plugin acc={r['plugin']['acc']*100:.2f}% "
                  f"nll={r['plugin']['nll']:.3f} ece={r['plugin']['ece']:.4f} "
                  f"({r['wall_clock_s']}s)", flush=True)
        except Exception as e:
            r = {'name': name, 'error': f'{type(e).__name__}: {e}',
                 'traceback': traceback.format_exc()}
            print(r['traceback'], flush=True)
        runs.append(r)

    out = {'meta': {'dataset': args.dataset, 'N': N, 'n_classes': n_classes,
                    'vit': {'dim': args.dim, 'depth': args.depth, 'heads': args.heads,
                            'patch': args.patch, 'params': n_params(model0)},
                    'lr': args.lr, 'adamw_lr': args.adamw_lr, 'hess_init': args.hess_init,
                    'wd': args.wd, 'clip_radius': args.clip_radius, 'epochs': args.epochs,
                    'jax': jax.__version__, 'devices': [str(d) for d in jax.devices()]},
           'runs': runs}
    with open(args.out, 'w') as f:
        json.dump(out, f, indent=2)

    print(f"\n=== ViT on {args.dataset} ({n_classes} classes) ===")
    for r in runs:
        if 'error' in r:
            print(f"{r['name']:7s} | ERROR: {r['error']}")
            continue
        p = r['plugin']
        line = f"{r['name']:7s} | mean acc {p['acc']*100:5.2f}%  nll {p['nll']:.3f}  ece {p['ece']:.4f}"
        if 'bma' in r:
            b = r['bma']
            line += f"  ||  BMA acc {b['acc']*100:5.2f}%  nll {b['nll']:.3f}  ece {b['ece']:.4f}"
        line += f"  ||  {r['wall_clock_s']}s"
        print(line)
    print('wrote', args.out, flush=True)


if __name__ == '__main__':
    main()
