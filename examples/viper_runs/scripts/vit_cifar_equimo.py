"""equimo ViT on CIFAR-100 from scratch: does the EVON paper's b1/b2 recipe
(beta1=0.95, beta2=0.95) beat the MNIST-notebook recipe (beta2=0.9999) for a ViT?

Configs (all share lr/ess/hess_init/clip; only the optimizer + betas differ):
  adamw           - non-Bayesian baseline
  ivon            - diagonal, beta1=0.95 beta2=0.9999 (paper IVON)
  evon-b2-0.95    - structured, beta1=0.95 beta2=0.95  (paper ViT recipe)
  evon-b2-0.9999  - structured, beta1=0.95 beta2=0.9999 (slow-Hessian, my earlier run)

Model: equimo VisionTransformer (real ViT), sized for CIFAR (32px, patch 4).
Runs on Viper-GPU (AMD MI300A, ROCm). ROCm-safe harness: host shuffle, Python
step loop, chunked eval (the fused scan+eval graph segfaults the ROCm compiler).
"""
import argparse
import json
import os
import time
import traceback

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
from jax import nn, vmap
import equinox as eqx
import optax

from blrax import ivon, evon, sample_posterior, noisy_value_and_grad
from blrax.states import ScaleByEvonState
from equimo.vision.models.vit import VisionTransformer

IS = eqx.is_inexact_array


def load_cifar(root, which='cifar100'):
    d = np.load(os.path.join(root, f'{which}.npz'))
    tr = d['train_x'].astype(np.float32) / 255.0
    te = d['test_x'].astype(np.float32) / 255.0
    mean = tr.reshape(-1, 3).mean(0)
    std = tr.reshape(-1, 3).std(0)
    tr = ((tr - mean) / std).transpose(0, 3, 1, 2)          # HWC -> CHW for equimo
    te = ((te - mean) / std).transpose(0, 3, 1, 2)
    return (jnp.asarray(tr), jnp.asarray(d['train_y'], jnp.int32),
            jnp.asarray(te), jnp.asarray(d['test_y'], jnp.int32), int(d['n_classes']))


def augment(key, imgs):
    """Standard CIFAR train aug on CHW batches: random 32-crop from a 4-pad + h-flip."""
    kf, kx, ky = jr.split(key, 3)
    B = imgs.shape[0]
    flip = jr.bernoulli(kf, 0.5, (B,))
    imgs = jnp.where(flip[:, None, None, None], imgs[:, :, :, ::-1], imgs)
    p = jnp.pad(imgs, ((0, 0), (0, 0), (4, 4), (4, 4)))
    ox = jr.randint(kx, (B,), 0, 9)
    oy = jr.randint(ky, (B,), 0, 9)
    return jax.vmap(lambda im, x, y: jax.lax.dynamic_slice(im, (0, x, y), (im.shape[0], 32, 32)))(p, ox, oy)


def build_model(key, n_classes, dim=192, depth=6, heads=3, patch=4):
    return VisionTransformer(
        img_size=32, in_channels=3, dim=dim, patch_size=patch,
        num_heads=heads, depths=[depth], num_classes=n_classes,
        reg_tokens=0, class_token=True, global_pool='token',
        drop_path_rate=0.0, pos_drop_rate=0.0, attn_drop=0.0, proj_drop=0.0,
        qk_norm=True, mlp_ratio=4.0, key=key)


def n_params(model):
    p, _ = eqx.partition(model, IS)
    return int(sum(x.size for x in jax.tree_util.tree_leaves(p)))


def _apply(model, x):                                        # x: (B,3,32,32) -> (B,C) logits
    return vmap(lambda img: model(img, inference=True))(x)


def run_training(key, model, optim, train_x, train_y, test_x, test_y, *,
                 epochs, batch, estimator, aug=True):
    params, static = eqx.partition(model, IS)
    opt_state = optim.init(params)
    N = int(train_x.shape[0])
    steps = N // batch

    def loss_fn(params, x, y, *args):
        return optax.softmax_cross_entropy_with_integer_labels(
            _apply(eqx.combine(params, static), x), y).mean()

    @jax.jit
    def train_step(params, opt_state, x, y, key):
        ak, sk = jr.split(key)
        if aug:
            x = augment(ak, x)
        l, g, s = noisy_value_and_grad(loss_fn, opt_state, params, sk, x, y, estimator=estimator)
        u, s = optim.update(g, s, params)
        return optax.apply_updates(params, u), s, jnp.mean(l)

    @jax.jit
    def eval_chunk(params, x):
        return _apply(eqx.combine(params, static), x)

    def evaluate(params):
        logits = jnp.concatenate(
            [eval_chunk(params, test_x[i:i + 1000]) for i in range(0, test_x.shape[0], 1000)], 0)
        acc = (logits.argmax(-1) == test_y).mean()
        nll = optax.softmax_cross_entropy_with_integer_labels(logits, test_y).mean()
        p = nn.softmax(logits, -1)
        conf = p.max(-1)
        pred = p.argmax(-1)
        bins = jnp.linspace(0, 1, 21)
        ece = 0.0
        for i in range(20):
            m = (conf > bins[i]) & (conf <= bins[i + 1])
            cnt = m.sum()
            ece = ece + jnp.where(cnt > 0,
                                  jnp.abs((pred == test_y)[m].sum() / jnp.maximum(cnt, 1)
                                          - conf[m].sum() / jnp.maximum(cnt, 1)) * cnt, 0.0)
        return float(acc), float(nll), float(ece / test_y.shape[0])

    rng = np.random.default_rng(0)
    curve = {'acc': [], 'nll': [], 'ece': [], 'loss': []}
    for _ in range(epochs):
        perm = rng.permutation(N)
        ep = jnp.zeros(())
        for s in range(steps):
            b = perm[s * batch:(s + 1) * batch]
            key, sk = jr.split(key)
            params, opt_state, l = train_step(params, opt_state, train_x[b], train_y[b], sk)
            ep = ep + l
        acc, nll, ece = evaluate(params)
        curve['loss'].append(float(ep) / steps)
        curve['acc'].append(acc); curve['nll'].append(nll); curve['ece'].append(ece)
    return eqx.combine(params, static), opt_state, {k: jnp.asarray(v) for k, v in curve.items()}


def bma_eval(key, model, opt_state, test_x, test_y, num_mc, n_classes):
    params, static = eqx.partition(model, IS)
    state0 = opt_state[0]
    out = sample_posterior(key, params, state0, shape=(num_mc,))
    samples = out if isinstance(state0, ScaleByEvonState) else out[0]

    @jax.jit
    def pred(sp, x):
        return nn.softmax(_apply(eqx.combine(sp, static), x), -1)

    probs = jnp.zeros((test_x.shape[0], n_classes))
    for i in range(num_mc):
        si = jax.tree.map(lambda a: a[i], samples)
        probs = probs + jnp.concatenate(
            [pred(si, test_x[j:j + 1000]) for j in range(0, test_x.shape[0], 1000)], 0)
    probs = probs / num_mc
    idx = jnp.arange(test_y.shape[0])
    conf = probs.max(-1); pr = probs.argmax(-1)
    bins = jnp.linspace(0, 1, 21); ece = 0.0
    for i in range(20):
        m = (conf > bins[i]) & (conf <= bins[i + 1]); cnt = m.sum()
        ece = ece + jnp.where(cnt > 0, jnp.abs((pr == test_y)[m].sum() / jnp.maximum(cnt, 1)
                              - conf[m].sum() / jnp.maximum(cnt, 1)) * cnt, 0.0)
    return {'acc': float((pr == test_y).mean()),
            'nll': float(-jnp.log(probs[idx, test_y] + 1e-12).mean()),
            'ece': float(ece / test_y.shape[0]), 'num_mc': num_mc}


def run_one(name, make_optim, estimator, key, model0, data, *, epochs, batch, num_mc, bayes, aug):
    train_x, train_y, test_x, test_y, n_classes = data
    k_tr, k_bma = jr.split(key)
    t0 = time.time()
    model, opt_state, metrics = run_training(
        k_tr, model0, make_optim(), train_x, train_y, test_x, test_y,
        epochs=epochs, batch=batch, estimator=estimator, aug=aug)
    jax.block_until_ready(metrics)
    wall = time.time() - t0
    c = {k: [float(v) for v in metrics[k]] for k in ('acc', 'nll', 'ece', 'loss')}
    res = {'name': name, 'wall_clock_s': round(wall, 1), 'n_params': n_params(model0),
           'plugin': {'acc': c['acc'][-1], 'nll': c['nll'][-1], 'ece': c['ece'][-1]}, 'curve': c}
    if bayes:
        res['bma'] = bma_eval(k_bma, model, opt_state, test_x, test_y, num_mc, n_classes)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='cifar100')
    ap.add_argument('--data-root', default='/ptmp/cheins/data')
    ap.add_argument('--out', default='results_vit_equimo.json')
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--augment', type=int, default=1)
    ap.add_argument('--batch', type=int, default=128)
    ap.add_argument('--dim', type=int, default=192)
    ap.add_argument('--depth', type=int, default=6)
    ap.add_argument('--heads', type=int, default=3)
    ap.add_argument('--lr', type=float, default=1e-2)        # Bayesian peak lr
    ap.add_argument('--adamw-lr', type=float, default=1e-3)
    ap.add_argument('--hess-init', type=float, default=0.35)
    ap.add_argument('--clip-radius', type=float, default=1e-1)
    ap.add_argument('--wd', type=float, default=1e-4)
    ap.add_argument('--num-mc', type=int, default=32)
    args = ap.parse_args()

    print('jax', jax.__version__, '| devices', jax.devices(), flush=True)
    data = load_cifar(args.data_root, args.dataset)
    N = int(data[0].shape[0]); n_classes = data[4]
    total = (N // args.batch) * args.epochs
    warm = max(1, total // 20)
    model0 = build_model(jr.PRNGKey(0), n_classes, dim=args.dim, depth=args.depth, heads=args.heads)
    print(f'equimo ViT dim={args.dim} depth={args.depth} heads={args.heads} | '
          f'params={n_params(model0):,} | {args.dataset} N={N} classes={n_classes} '
          f'| steps={total} warmup={warm}', flush=True)

    def sch(peak):
        return optax.warmup_cosine_decay_schedule(0.0, peak, warm, total, end_value=peak * 0.1)

    aug = bool(args.augment)
    keys = jr.split(jr.PRNGKey(7), 3)
    specs = [
        ('adamw', lambda: optax.adamw(sch(args.adamw_lr), weight_decay=0.05), 'sampling', keys[0], False),
        ('ivon', lambda: ivon(sch(args.lr), ess=N, hess_init=args.hess_init, weight_decay=args.wd,
                              clip_radius=args.clip_radius, b1=0.95, b2=0.9999), 'sampling', keys[1], True),
        ('evon', lambda: evon(sch(args.lr), ess=N, hess_init=args.hess_init, weight_decay=args.wd,
                              clip_radius=args.clip_radius, precond_every=10, max_precond_dim=10000,
                              one_sided=False, b1=0.95, b2=0.9999, b3=0.95), 'sampling', keys[2], True),
    ]

    runs = []
    for name, mk, est, key, bayes in specs:
        print(f'--- running {name} (aug={aug}) ---', flush=True)
        try:
            r = run_one(name, mk, est, key, model0, data,
                        epochs=args.epochs, batch=args.batch, num_mc=args.num_mc, bayes=bayes, aug=aug)
            print(f"    done: acc={r['plugin']['acc']*100:.2f}% nll={r['plugin']['nll']:.3f} "
                  f"ece={r['plugin']['ece']:.4f} ({r['wall_clock_s']}s)", flush=True)
        except Exception as e:
            r = {'name': name, 'error': f'{type(e).__name__}: {e}', 'traceback': traceback.format_exc()}
            print(r['traceback'], flush=True)
        runs.append(r)

    out = {'meta': {'dataset': args.dataset, 'N': N, 'n_classes': n_classes,
                    'model': 'equimo VisionTransformer', 'dim': args.dim, 'depth': args.depth,
                    'params': n_params(model0), 'lr': args.lr, 'hess_init': args.hess_init,
                    'clip_radius': args.clip_radius, 'wd': args.wd, 'epochs': args.epochs,
                    'augment': aug, 'jax': jax.__version__,
                    'devices': [str(d) for d in jax.devices()]}, 'runs': runs}
    with open(args.out, 'w') as f:
        json.dump(out, f, indent=2)

    print(f"\n=== equimo ViT on {args.dataset} — IVON vs EVON | aug={aug} | {args.epochs} epochs ===")
    for r in runs:
        if 'error' in r:
            print(f"{r['name']:15s} | ERROR: {r['error']}"); continue
        p = r['plugin']
        line = f"{r['name']:15s} | acc {p['acc']*100:5.2f}%  nll {p['nll']:.3f}  ece {p['ece']:.4f}"
        if 'bma' in r:
            b = r['bma']
            line += f"  ||  BMA acc {b['acc']*100:5.2f}%  nll {b['nll']:.3f}  ece {b['ece']:.4f}"
        line += f"  ||  {r['wall_clock_s']}s"
        print(line)
    print('wrote', args.out, flush=True)


if __name__ == '__main__':
    main()
