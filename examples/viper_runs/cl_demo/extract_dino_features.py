"""Extract frozen DINOv2 ViT-S/14 CLS features for CIFAR-100 (one-off).

Reads /ptmp/cheins/data/cifar100.npz (uint8 HWC), resizes 32->224 (bicubic),
normalizes with ImageNet stats, runs equimo dinov2_vits14(pretrained=True,
dynamic_img_size=True) and saves x_norm_cls_token features (384-d).

Warm the weight cache on the LOGIN node first (shared $HOME):
  python -c "from equimo.vision.models.vit import dinov2_vits14 as f; f(pretrained=True, dynamic_img_size=True)"
"""
import argparse
import time

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx

from equimo.vision.models.vit import dinov2_vits14

MEAN = jnp.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
STD = jnp.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cifar', default='/ptmp/cheins/data/cifar100.npz')
    ap.add_argument('--out', default='/ptmp/cheins/data/cifar100_dinov2s14.npz')
    ap.add_argument('--batch', type=int, default=256)
    args = ap.parse_args()

    print('jax', jax.__version__, '| devices', jax.devices(), flush=True)
    model = dinov2_vits14(pretrained=True, dynamic_img_size=True)
    print('backbone loaded', flush=True)

    @eqx.filter_jit
    def embed(batch_u8):                       # (B, 32, 32, 3) uint8
        x = batch_u8.astype(jnp.float32) / 255.0
        x = jnp.transpose(x, (0, 3, 1, 2))     # -> (B, 3, 32, 32)
        x = jax.image.resize(x, (x.shape[0], 3, 224, 224), method='bicubic')
        x = (x - MEAN) / STD

        def one(img):
            return model.forward_features(img, key=jr.PRNGKey(0),
                                          inference=True)['x_norm_cls_token']
        return jax.vmap(one)(x)

    d = np.load(args.cifar)
    out = {}
    for split in ('train', 'test'):
        X = d[f'{split}_x']; y = d[f'{split}_y']
        feats = []
        t0 = time.time()
        for i in range(0, len(X), args.batch):
            feats.append(np.asarray(embed(jnp.asarray(X[i:i + args.batch]))))
        F = np.concatenate(feats, 0)
        print(f'{split}: {F.shape} in {time.time()-t0:.1f}s', flush=True)
        out[f'{split}_x'] = F.astype(np.float32)
        out[f'{split}_y'] = y.astype(np.int32)

    np.savez(args.out, **out)
    print('wrote', args.out, flush=True)


if __name__ == '__main__':
    main()
