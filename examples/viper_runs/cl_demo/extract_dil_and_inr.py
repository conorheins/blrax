"""Feature extraction for the two extra CL benchmarks (one-off, Viper-GPU):

  rotated-CIFAR-10 DIL: 5 domains (0/22.5/45/67.5/90 deg rotations of CIFAR-10),
    same 10-way label space -> /ptmp/cheins/data/cifar10_rot_dinov2s14.npz
    with train_x/train_y/train_dom + test_* (features 384-d).

  Split ImageNet-R: 200 classes, 80/20 per-class train/test split (seed 0)
    -> /ptmp/cheins/data/imagenet_r_dinov2s14.npz.

Backbone: equimo dinov2_vits14(pretrained=True, dynamic_img_size=True), CLS token.
"""
import argparse
import os
import time

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from PIL import Image

from equimo.vision.models.vit import dinov2_vits14

MEAN = jnp.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
STD = jnp.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)


def make_embed(model, in_hw):
    @eqx.filter_jit
    def embed(batch_u8):                      # (B, H, W, 3) uint8
        x = batch_u8.astype(jnp.float32) / 255.0
        x = jnp.transpose(x, (0, 3, 1, 2))
        if in_hw != 224:
            x = jax.image.resize(x, (x.shape[0], 3, 224, 224), method='bicubic')
        x = (x - MEAN) / STD

        def one(img):
            return model.forward_features(img, key=jr.PRNGKey(0),
                                          inference=True)['x_norm_cls_token']
        return jax.vmap(one)(x)
    return embed


def embed_all(embed, X, batch):
    out = []
    for i in range(0, len(X), batch):
        out.append(np.asarray(embed(jnp.asarray(X[i:i + batch]))))
    return np.concatenate(out, 0).astype(np.float32)


def do_rotated_cifar(model, args):
    d = np.load('/ptmp/cheins/data/cifar10.npz')
    angles = [0.0, 22.5, 45.0, 67.5, 90.0]
    embed = make_embed(model, 32)
    out = {}
    for split in ('train', 'test'):
        X, y = d[f'{split}_x'], d[f'{split}_y']
        fs, ys, ds = [], [], []
        for di, ang in enumerate(angles):
            t0 = time.time()
            if ang == 0.0:
                Xr = X
            else:
                Xr = np.stack([np.asarray(
                    Image.fromarray(im).rotate(ang, resample=Image.BILINEAR,
                                               fillcolor=(114, 114, 114)))
                    for im in X])
            fs.append(embed_all(embed, Xr, args.batch))
            ys.append(y); ds.append(np.full(len(y), di, np.int32))
            print(f'rotcifar {split} dom{di} ({ang} deg): {time.time()-t0:.1f}s', flush=True)
        out[f'{split}_x'] = np.concatenate(fs, 0)
        out[f'{split}_y'] = np.concatenate(ys, 0).astype(np.int32)
        out[f'{split}_dom'] = np.concatenate(ds, 0)
    np.savez('/ptmp/cheins/data/cifar10_rot_dinov2s14.npz', **out)
    print('wrote cifar10_rot_dinov2s14.npz', {k: v.shape for k, v in out.items()}, flush=True)


def load_center224(path):
    im = Image.open(path).convert('RGB')
    w, h = im.size
    s = 224 / min(w, h)
    im = im.resize((max(224, int(round(w * s))), max(224, int(round(h * s)))),
                   Image.BICUBIC)
    w, h = im.size
    l, t = (w - 224) // 2, (h - 224) // 2
    return np.asarray(im.crop((l, t, l + 224, t + 224)), np.uint8)


def do_imagenet_r(model, args):
    root = '/ptmp/cheins/data/imagenet-r'
    wnids = sorted(d for d in os.listdir(root)
                   if os.path.isdir(os.path.join(root, d)))
    assert len(wnids) == 200, f'expected 200 classes, got {len(wnids)}'
    embed = make_embed(model, 224)
    rng = np.random.default_rng(0)
    tr_f, tr_y, te_f, te_y = [], [], [], []
    for ci, wnid in enumerate(wnids):
        files = sorted(os.listdir(os.path.join(root, wnid)))
        imgs = np.stack([load_center224(os.path.join(root, wnid, f)) for f in files])
        F = embed_all(embed, imgs, args.batch)
        perm = rng.permutation(len(F))
        n_te = max(1, int(0.2 * len(F)))
        te, tr = perm[:n_te], perm[n_te:]
        tr_f.append(F[tr]); tr_y.append(np.full(len(tr), ci, np.int32))
        te_f.append(F[te]); te_y.append(np.full(len(te), ci, np.int32))
        if ci % 25 == 0:
            print(f'imagenet-r class {ci}/200 ({len(F)} imgs)', flush=True)
    np.savez('/ptmp/cheins/data/imagenet_r_dinov2s14.npz',
             train_x=np.concatenate(tr_f), train_y=np.concatenate(tr_y),
             test_x=np.concatenate(te_f), test_y=np.concatenate(te_y))
    print('wrote imagenet_r_dinov2s14.npz | train',
          sum(len(a) for a in tr_f), 'test', sum(len(a) for a in te_f), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--which', choices=['rotcifar', 'inr', 'both'], default='both')
    ap.add_argument('--batch', type=int, default=256)
    args = ap.parse_args()
    print('jax', jax.__version__, '| devices', jax.devices(), flush=True)
    model = dinov2_vits14(pretrained=True, dynamic_img_size=True)
    print('backbone loaded', flush=True)
    if args.which in ('rotcifar', 'both'):
        do_rotated_cifar(model, args)
    if args.which in ('inr', 'both'):
        do_imagenet_r(model, args)


if __name__ == '__main__':
    main()
