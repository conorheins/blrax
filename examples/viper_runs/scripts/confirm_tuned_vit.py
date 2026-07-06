"""Step 2 of the tuning protocol: long run (60 epochs, CIFAR-100 + aug) with the
Optuna-tuned configs for AdamW / IVON / EVON, read directly from the sweep JSON
(optuna_vit_results.json best_params) so nothing is retyped by hand.

Same harness/eval as vit_cifar_equimo.py (ROCm-safe, plugin + BMA metrics).
"""
import argparse
import json
import time
import traceback

import jax
import jax.numpy as jnp
import jax.random as jr
import optax

from blrax import ivon, evon
from vit_cifar_equimo import load_cifar, build_model, n_params, run_one


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sweep-json', default='/ptmp/cheins/blrax/examples/optuna_vit_results.json')
    ap.add_argument('--data-root', default='/ptmp/cheins/data')
    ap.add_argument('--dataset', default='cifar100')
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--batch', type=int, default=128)
    ap.add_argument('--dim', type=int, default=192)
    ap.add_argument('--depth', type=int, default=6)
    ap.add_argument('--heads', type=int, default=3)
    ap.add_argument('--num-mc', type=int, default=32)
    ap.add_argument('--out', default='results_vit_tuned.json')
    args = ap.parse_args()

    best = {k: v['best_params'] for k, v in
            json.load(open(args.sweep_json))['studies'].items()}
    print('tuned params:', json.dumps(best, indent=1), flush=True)

    print('jax', jax.__version__, '| devices', jax.devices(), flush=True)
    data = load_cifar(args.data_root, args.dataset)
    N = int(data[0].shape[0])
    total = (N // args.batch) * args.epochs
    warm = max(1, total // 20)
    model0 = build_model(jr.PRNGKey(0), data[4], dim=args.dim, depth=args.depth,
                         heads=args.heads)
    print(f'equimo ViT params={n_params(model0):,} | steps={total} warmup={warm}', flush=True)

    def sch(peak):
        return optax.warmup_cosine_decay_schedule(0.0, peak, warm, total,
                                                  end_value=peak * 0.1)

    a, i, e = best['adamw'], best['ivon'], best['evon']
    specs = [
        ('adamw*', lambda: optax.adamw(sch(a['lr']), weight_decay=a['wd']),
         'sampling', False),
        ('ivon*', lambda: ivon(sch(i['lr']), ess=N * i['ess_mult'],
                               hess_init=i['hess_init'], weight_decay=i['wd'],
                               clip_radius=i['clip_radius'], b1=i['b1'], b2=i['b2']),
         'sampling', True),
        ('evon*', lambda: evon(sch(e['lr']), ess=N * e['ess_mult'],
                               hess_init=e['hess_init'], weight_decay=e['wd'],
                               clip_radius=e['clip_radius'], b1=e['b1'], b2=e['b2'],
                               b3=e['b3'], precond_every=10, max_precond_dim=10000,
                               one_sided=False),
         'sampling', True),
    ]

    runs = []
    keys = jr.split(jr.PRNGKey(7), len(specs))
    for (name, mk, est, bayes), key in zip(specs, keys):
        print(f'--- running {name} ---', flush=True)
        try:
            r = run_one(name, mk, est, key, model0, data, epochs=args.epochs,
                        batch=args.batch, num_mc=args.num_mc, bayes=bayes, aug=True)
            print(f"    done: acc={r['plugin']['acc']*100:.2f}% "
                  f"nll={r['plugin']['nll']:.3f} ece={r['plugin']['ece']:.4f} "
                  f"({r['wall_clock_s']}s)", flush=True)
        except Exception as ex:
            r = {'name': name, 'error': f'{type(ex).__name__}: {ex}',
                 'traceback': traceback.format_exc()}
            print(r['traceback'], flush=True)
        runs.append(r)

    out = {'meta': {'tuned_params': best, 'dataset': args.dataset, 'N': N,
                    'epochs': args.epochs, 'augment': True,
                    'params': n_params(model0), 'jax': jax.__version__},
           'runs': runs}
    with open(args.out, 'w') as f:
        json.dump(out, f, indent=2)

    print(f"\n=== TUNED long run: equimo ViT on {args.dataset}, {args.epochs} ep + aug ===")
    for r in runs:
        if 'error' in r:
            print(f"{r['name']:7s} | ERROR: {r['error']}")
            continue
        p = r['plugin']
        line = f"{r['name']:7s} | acc {p['acc']*100:5.2f}%  nll {p['nll']:.3f}  ece {p['ece']:.4f}"
        if 'bma' in r:
            b = r['bma']
            line += f"  ||  BMA acc {b['acc']*100:5.2f}%  nll {b['nll']:.3f}  ece {b['ece']:.4f}"
        line += f"  ||  {r['wall_clock_s']}s"
        print(line)
    print('wrote', args.out, flush=True)


if __name__ == '__main__':
    main()
