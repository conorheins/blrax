"""Continual learning with Bayesian LoRA finetuning of the DINOv2 backbone.

Same Split CIFAR-100 CIL protocol as cl_split_cifar.py, but instead of a
linear probe on frozen features, the trainable parameters are LoRA adapters
(rank 8 on attention qkv/proj via equimo.finetune) + a linear head; the VCL
posterior->prior recursion runs over that whole (LoRA + head) pytree.

Prior at task 1 pulls LoRA toward its init (lora_B = 0 -> the pretrained
function), so the base prior literally means "stay close to DINOv2".
"""
import argparse
import json
import time

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
from jax import nn, vmap
import equinox as eqx
import optax
from optax import tree_utils as otu

from blrax import ivon, evon, noisy_value_and_grad, get_scale, sample_posterior
from blrax.states import ScaleByIvonState, ScaleByEvonState, _is_evon_leaf
from equimo.vision.models.vit import dinov2_vits14
from equimo.finetune.recipes import lora_transformer

WD_EPS = 1e-7
NEG = -1e9
MEAN = jnp.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
STD = jnp.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)


def tree_sum(t):
    return sum(jnp.sum(l) for l in jtu.tree_leaves(t))


def make_tasks(train_y, test_y, n_tasks, seed):
    n_classes = int(train_y.max()) + 1
    order = np.random.default_rng(seed).permutation(n_classes)
    per = n_classes // n_tasks
    ytr, yte = np.asarray(train_y), np.asarray(test_y)
    return [{'classes': order[t*per:(t+1)*per],
             'train_idx': np.where(np.isin(ytr, order[t*per:(t+1)*per]))[0],
             'test_idx': np.where(np.isin(yte, order[t*per:(t+1)*per]))[0]}
            for t in range(n_tasks)], order


def build_model_and_partition(key, rank):
    mk, hk = jr.split(key)
    backbone = dinov2_vits14(pretrained=True, dynamic_img_size=True)
    backbone = lora_transformer(backbone, key=mk, rank=rank, alpha=2.0 * rank)

    def is_lora(path, leaf):
        return eqx.is_inexact_array(leaf) and 'lora_' in jtu.keystr(path).lower()
    filt = jtu.tree_map_with_path(is_lora, backbone)
    lora_params, static = eqx.partition(backbone, filt)
    n_lora = sum(x.size for x in jtu.tree_leaves(lora_params))
    head = {'w': 0.01 * jr.normal(hk, (backbone.dim, 100)), 'b': jnp.zeros(100)}
    print(f'LoRA params: {n_lora:,} | head: {backbone.dim*100+100:,}', flush=True)
    return {'lora': lora_params, 'head': head}, static


def make_apply(static):
    def apply(params, imgs_u8):                    # (B, 32, 32, 3) uint8 -> logits
        x = imgs_u8.astype(jnp.float32) / 255.0
        x = jnp.transpose(x, (0, 3, 1, 2))
        x = jax.image.resize(x, (x.shape[0], 3, 224, 224), method='bicubic')
        x = (x - MEAN) / STD
        model = eqx.combine(params['lora'], static)

        def one(img):
            return model.forward_features(img, key=jr.PRNGKey(0),
                                          inference=True)['x_norm_cls_token']
        feats = jax.vmap(one)(x)
        return feats @ params['head']['w'] + params['head']['b']
    return apply


def run_method(method, hp, Xtr, ytr, Xte, yte, tasks, static, params0, key, *,
               epochs, batch, num_mc):
    apply = make_apply(static)
    n_classes = 100
    params = params0
    bayes = method.startswith(('ivon', 'evon'))
    cl = method.endswith('-cl')

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

    mu_prev = lam_prev = None
    if cl:
        mu_prev = params0
        lam_prev = jtu.tree_map(lambda p: hp['p0'] * jnp.ones_like(p), params0)

    opt_state = None
    R, union, seen_np = [], [], np.array([], int)
    for t, task in enumerate(tasks):
        tr = task['train_idx']
        zeta = hp.get('tau', 1.0) * len(tr)
        seen_np = np.concatenate([seen_np, task['classes']]).astype(int)
        seen_mask = jnp.zeros(n_classes, bool).at[jnp.asarray(seen_np)].set(True)

        def loss_fn(params, x, y, key):
            lg = apply(params, x) + jnp.where(seen_mask, 0.0, NEG)
            ce = optax.softmax_cross_entropy_with_integer_labels(lg, y).mean()
            if cl:
                quad = 0.5 * tree_sum(jtu.tree_map(
                    lambda l, p, m: jnp.sum(l * (p - m) ** 2),
                    lam_prev, params, mu_prev))
                ce = ce + quad / zeta
            return ce

        optim = mk_optim(zeta)
        if opt_state is None or not (cl and bayes):
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

        @jax.jit
        def step(params, opt_state, x, y, k):
            if bayes:
                l, g, opt_state = noisy_value_and_grad(loss_fn, opt_state, params, k, x, y)
            else:
                l, g = jax.value_and_grad(loss_fn)(params, x, y, k)
            u, opt_state = optim.update(g, opt_state, params)
            return optax.apply_updates(params, u), opt_state, jnp.mean(l)

        rng = np.random.default_rng(0)
        N = len(tr)
        t0 = time.time()
        for _ in range(epochs):
            perm = rng.permutation(N)
            for s_ in range(max(1, N // batch)):
                b = tr[perm[s_ * batch:(s_ + 1) * batch]]
                key, k = jr.split(key)
                params, opt_state, _ = step(params, opt_state, Xtr[b], ytr[b], k)

        if cl:
            mu_prev = params
            lam_prev = jtu.tree_map(lambda s_: 1.0 / s_ ** 2, get_scale(opt_state[0]))

        # ---- eval on seen tasks ----
        @jax.jit
        def logits_of(params, x):
            return apply(params, x) + jnp.where(seen_mask, 0.0, NEG)

        def acc_on(idx):
            hits = 0
            for i in range(0, len(idx), 500):
                b = idx[i:i + 500]
                hits += int((logits_of(params, Xte[b]).argmax(-1) == yte[b]).sum())
            return hits / len(idx)

        row = [acc_on(tasks[j]['test_idx']) for j in range(t + 1)]
        uni_idx = np.concatenate([tasks[j]['test_idx'] for j in range(t + 1)])
        nll_s = 0.0; hits = 0
        for i in range(0, len(uni_idx), 500):
            b = uni_idx[i:i + 500]
            lg = logits_of(params, Xte[b])
            hits += int((lg.argmax(-1) == yte[b]).sum())
            nll_s += float(optax.softmax_cross_entropy_with_integer_labels(lg, yte[b]).sum())
        R.append(row)
        union.append({'acc': hits / len(uni_idx), 'nll': nll_s / len(uni_idx)})
        print(f'  task {t+1}: union acc {union[-1]["acc"]*100:.2f}% '
              f'({time.time()-t0:.0f}s)', flush=True)

    # final BMA
    bma = None
    if bayes:
        key, bk = jr.split(key)
        out = sample_posterior(bk, params, opt_state[0], shape=(num_mc,))
        samples = out if isinstance(opt_state[0], ScaleByEvonState) else out[0]
        uni_idx = np.concatenate([t_['test_idx'] for t_ in tasks])
        probs = None
        for i in range(num_mc):
            si = jtu.tree_map(lambda a: a[i], samples)
            ps = []
            for j in range(0, len(uni_idx), 500):
                b = uni_idx[j:j + 500]
                ps.append(nn.softmax(apply(si, Xte[b])
                                     + jnp.where(seen_mask, 0.0, NEG), -1))
            p = jnp.concatenate(ps, 0)
            probs = p if probs is None else probs + p
        probs = probs / num_mc
        y = yte[jnp.asarray(uni_idx)]
        bma = {'acc': float((probs.argmax(-1) == y).mean()),
               'nll': float(-jnp.log(probs[jnp.arange(len(uni_idx)), y] + 1e-12).mean()),
               'num_mc': num_mc}

    T = len(tasks)
    metrics = {'ACC': float(np.mean(R[-1])),
               'AIA': float(np.mean([np.mean(r) for r in R])),
               'forgetting': float(np.mean([max(R[tt][j] for tt in range(j, T)) - R[-1][j]
                                            for j in range(T - 1)])),
               'LA': float(np.mean([R[tt][tt] for tt in range(T)])),
               'union_final': union[-1], 'bma_final': bma}
    return {'method': method, 'hp': hp, 'R': R, 'union': union, 'metrics': metrics}


DEFAULT_HP = {
    'adamw':   {'lr': 1e-3, 'wd': 0.0},
    'ivon-cl': {'lr': 0.03, 'h0': 0.5, 'b2': 0.99999, 'p0': 1.0, 'tau': 10.0},
    'evon-cl': {'lr': 0.03, 'h0': 0.5, 'b2': 0.99999, 'p0': 1.0, 'tau': 10.0},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cifar', default='/ptmp/cheins/data/cifar100.npz')
    ap.add_argument('--methods', default='adamw,ivon-cl')
    ap.add_argument('--n-tasks', type=int, default=10)
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--batch', type=int, default=128)
    ap.add_argument('--rank', type=int, default=8)
    ap.add_argument('--num-mc', type=int, default=8)
    ap.add_argument('--seeds', default='0')
    ap.add_argument('--hp', default='{}')
    ap.add_argument('--out', default='results_cl_lora.json')
    args = ap.parse_args()

    print('jax', jax.__version__, '| devices', jax.devices(), flush=True)
    d = np.load(args.cifar)
    Xtr = jnp.asarray(d['train_x']); ytr = jnp.asarray(d['train_y'], jnp.int32)
    Xte = jnp.asarray(d['test_x']); yte = jnp.asarray(d['test_y'], jnp.int32)

    overrides = json.loads(args.hp)
    runs = []
    for seed in [int(s) for s in args.seeds.split(',')]:
        tasks, _ = make_tasks(d['train_y'], d['test_y'], args.n_tasks, seed)
        params0, static = build_model_and_partition(jr.PRNGKey(100 + seed), args.rank)
        for method in args.methods.split(','):
            hp = dict(DEFAULT_HP[method]); hp.update(overrides.get(method, {}))
            print(f'--- {method} seed{seed} hp={hp} ---', flush=True)
            t0 = time.time()
            r = run_method(method, hp, Xtr, ytr, Xte, yte, tasks, static, params0,
                           jr.PRNGKey(1000 + seed), epochs=args.epochs,
                           batch=args.batch, num_mc=args.num_mc)
            r['seed'] = seed; r['wall_s'] = round(time.time() - t0, 1)
            m = r['metrics']
            line = (f"seed{seed} {method:8s} | ACC {m['ACC']*100:5.2f}%  "
                    f"AIA {m['AIA']*100:5.2f}%  forget {m['forgetting']*100:5.2f}  "
                    f"LA {m['LA']*100:5.2f} | union acc {m['union_final']['acc']*100:5.2f}% "
                    f"nll {m['union_final']['nll']:.3f}")
            if m['bma_final']:
                line += f" | BMA acc {m['bma_final']['acc']*100:.2f}%"
            print(line + f" ({r['wall_s']}s)", flush=True)
            runs.append(r)

    with open(args.out, 'w') as f:
        json.dump({'meta': vars(args), 'runs': runs}, f, indent=2, default=str)
    print('wrote', args.out, flush=True)


if __name__ == '__main__':
    main()
