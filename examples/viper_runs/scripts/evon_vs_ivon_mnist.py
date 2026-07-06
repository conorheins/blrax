"""EVON vs IVON vs AdamW on MNIST (MLP), matching the repo's
compare_ivon_evon.ipynb recipe (lr=1e-2, hess_init=0.35, weight_decay=1e-4,
clip_radius=1e-1, ess=N).

Reports point-estimate (mean) and Bayesian-model-averaging (BMA) acc/NLL/ECE.
Includes h-evon (the at-the-mean Hutchinson EVON estimator). The EVON runs
exercise the eigenbasis refresh / power-iteration path that commit 91be154
(diagonal-Hessian re-sort) fixed.
"""
import argparse
import json
import time
import traceback

import jax
import jax.numpy as jnp
import jax.random as jr
from jax import nn, vmap
import equinox as eqx
import optax

from blrax import ivon, evon, sample_posterior
from blrax.states import ScaleByEvonState
from data_loaders import load_mnist
from training import run_training, compute_ece

IN, OUT = 28 * 28, 10


def make_model(key, width, depth):
    return eqx.nn.MLP(IN, OUT, width, depth, key=key)


def n_params(model):
    p, _ = eqx.partition(model, eqx.is_array)
    return int(sum(x.size for x in jax.tree_util.tree_leaves(p)))


def plugin_eval(model, test_x, test_y):
    logits = vmap(model)(test_x)
    acc = (logits.argmax(-1) == test_y).mean()
    nll = optax.softmax_cross_entropy_with_integer_labels(logits, test_y).mean()
    return float(acc), float(nll), float(compute_ece(logits, test_y))


def bma_eval(key, model, opt_state, test_x, test_y, num_mc):
    """Average softmax over posterior weight samples (python loop = flat memory)."""
    params, static = eqx.partition(model, eqx.is_array)
    state0 = opt_state[0]
    out = sample_posterior(key, params, state0, shape=(num_mc,))
    samples = out if isinstance(state0, ScaleByEvonState) else out[0]
    probs = jnp.zeros((test_x.shape[0], OUT))
    for i in range(num_mc):
        si = jax.tree.map(lambda a: a[i], samples)
        probs = probs + nn.softmax(vmap(eqx.combine(si, static))(test_x), -1)
    probs = probs / num_mc
    logp = jnp.log(probs + 1e-12)
    idx = jnp.arange(test_y.shape[0])
    return {'acc': float((probs.argmax(-1) == test_y).mean()),
            'nll': float(-logp[idx, test_y].mean()),
            'ece': float(compute_ece(logp, test_y)), 'num_mc': num_mc}


def run_one(name, make_optim, estimator, key, model0, train_ds, test_ds,
            *, epochs, batch, num_mc, bayes):
    k_train, k_bma = jr.split(key)
    t0 = time.time()
    model, opt_state, metrics = run_training(
        k_train, model0, make_optim(), train_ds, test_ds,
        num_epochs=epochs, batch_size=batch, estimator=estimator)
    jax.block_until_ready(metrics)
    wall = time.time() - t0
    test_x = test_ds['image'].reshape(-1, IN)
    test_y = test_ds['label']
    acc, nll, ece = plugin_eval(model, test_x, test_y)
    res = {'name': name, 'estimator': estimator, 'wall_clock_s': round(wall, 2),
           'n_params': n_params(model0), 'epochs': epochs,
           'plugin': {'acc': acc, 'nll': nll, 'ece': ece},
           'curve': {k: [float(v) for v in metrics[k]] for k in ('acc', 'nll', 'ece', 'loss')}}
    if bayes:
        res['bma'] = bma_eval(k_bma, model, opt_state, test_x, test_y, num_mc)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='results_evon_vs_ivon_v3.json')
    ap.add_argument('--epochs', type=int, default=15)
    ap.add_argument('--batch', type=int, default=128)
    ap.add_argument('--width', type=int, default=512)
    ap.add_argument('--depth', type=int, default=3)
    ap.add_argument('--lr', type=float, default=1e-2)
    ap.add_argument('--hess-init', type=float, default=0.35)
    ap.add_argument('--wd', type=float, default=1e-4)
    ap.add_argument('--clip-radius', type=float, default=1e-1)
    ap.add_argument('--adamw-lr', type=float, default=1e-3)
    ap.add_argument('--num-mc', type=int, default=32)
    args = ap.parse_args()

    print('jax', jax.__version__, '| devices', jax.devices(), flush=True)
    train_ds, test_ds = load_mnist()
    N = int(train_ds['image'].shape[0])
    model0 = make_model(jr.PRNGKey(0), args.width, args.depth)   # identical init for all
    keys = dict(zip(('adamw', 'ivon', 'evon', 'h-evon'), jr.split(jr.PRNGKey(42), 4)))

    def mk_evon():
        return evon(args.lr, ess=N, hess_init=args.hess_init, weight_decay=args.wd,
                    clip_radius=args.clip_radius, precond_every=10, max_precond_dim=10000,
                    one_sided=False, b2=0.9999)

    specs = [
        ('adamw', lambda: optax.adamw(args.adamw_lr, weight_decay=args.wd), 'sampling',
         keys['adamw'], False),
        ('ivon', lambda: ivon(args.lr, ess=N, hess_init=args.hess_init, weight_decay=args.wd,
                              clip_radius=args.clip_radius), 'sampling', keys['ivon'], True),
        ('evon', mk_evon, 'sampling', keys['evon'], True),
        ('h-evon', mk_evon, 'hutchinson', keys['h-evon'], True),
    ]

    runs = []
    for name, mk, est, key, bayes in specs:
        print(f'--- running {name} ({est}) ---', flush=True)
        try:
            r = run_one(name, mk, est, key, model0, train_ds, test_ds,
                        epochs=args.epochs, batch=args.batch, num_mc=args.num_mc, bayes=bayes)
        except Exception as e:
            r = {'name': name, 'error': f'{type(e).__name__}: {e}',
                 'traceback': traceback.format_exc()}
            print(r['traceback'], flush=True)
        runs.append(r)

    out = {'meta': {'N': N, 'width': args.width, 'depth': args.depth, 'lr': args.lr,
                    'hess_init': args.hess_init, 'wd': args.wd, 'clip_radius': args.clip_radius,
                    'adamw_lr': args.adamw_lr, 'jax': jax.__version__,
                    'devices': [str(d) for d in jax.devices()]}, 'runs': runs}
    with open(args.out, 'w') as f:
        json.dump(out, f, indent=2)

    print(f"\n=== EVON vs IVON vs AdamW on MNIST (MLP {args.width}x{args.depth}) ===")
    print(f"lr={args.lr} ess=N={N} hess_init={args.hess_init} wd={args.wd} "
          f"clip_radius={args.clip_radius} epochs={args.epochs}")
    for r in runs:
        if 'error' in r:
            print(f"{r['name']:7s} | ERROR: {r['error']}")
            continue
        p = r['plugin']
        line = f"{r['name']:7s} | mean acc {p['acc']*100:5.2f}%  nll {p['nll']:.4f}  ece {p['ece']:.4f}"
        if 'bma' in r:
            b = r['bma']
            line += f"  ||  BMA acc {b['acc']*100:5.2f}%  nll {b['nll']:.4f}  ece {b['ece']:.4f}"
        line += f"  ||  {r['wall_clock_s']:.1f}s"
        print(line)
    print('wrote', args.out, flush=True)


if __name__ == '__main__':
    main()
