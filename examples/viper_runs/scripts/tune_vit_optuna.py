"""Short-run Optuna sweeps for IVON/EVON on the equimo ViT (CIFAR-100 + aug).

Debugging protocol (Dimitrije): the Bayesian methods lag tuned AdamW badly in
the FIRST epochs (aug run: AdamW ~21% @ epoch 2 vs IVON/EVON ~4%), which smells
like one hyperparameter is off. So: many very short runs, objective = accuracy
after 2 epochs, sweep the suspects (hess_init, b3, betas incl. b2=0.95, ess/zeta,
lr, clip_radius, wd), find the regime that closes the early gap, THEN do a long
run with the winners.

Conventions (mirrors examples/tune_ivon.py): fixed model init + fixed training
key across trials -> low-variance objective for TPE. Results JSON is rewritten
after every trial so partial sweeps survive job death. jax.clear_caches()
between trials keeps the XLA cache from growing across ~75 recompiles.

Runs on Viper-GPU (ROCm-safe harness: python step loop, chunked eval).
"""
import argparse
import json
import os
import time

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
from jax import nn, vmap
import equinox as eqx
import optax
import optuna

from blrax import ivon, evon, noisy_value_and_grad
from vit_cifar_equimo import load_cifar, build_model, augment, _apply

IS = eqx.is_inexact_array


# ---------------------------------------------------------------- short train
def make_short_trainer(model0, data, *, epochs, batch, eval_n):
    train_x, train_y, test_x, test_y, n_classes = data
    ev_x, ev_y = test_x[:eval_n], test_y[:eval_n]
    params0, static = eqx.partition(model0, IS)
    N = int(train_x.shape[0])
    steps = N // batch

    def loss_fn(params, x, y, *args):
        return optax.softmax_cross_entropy_with_integer_labels(
            _apply(eqx.combine(params, static), x), y).mean()

    def run(optim, estimator, report=None):
        """Train `epochs` epochs; return list of per-epoch (acc, nll) on the
        eval subset. Bails out early (returns what it has) on NaN."""
        @jax.jit
        def train_step(params, opt_state, x, y, key):
            ak, sk = jr.split(key)
            x = augment(ak, x)
            l, g, s = noisy_value_and_grad(loss_fn, opt_state, params, sk, x, y,
                                           estimator=estimator)
            u, s = optim.update(g, s, params)
            return optax.apply_updates(params, u), s, jnp.mean(l)

        @jax.jit
        def eval_subset(params):
            logits = jnp.concatenate(
                [_apply(eqx.combine(params, static), ev_x[i:i + 1000])
                 for i in range(0, ev_x.shape[0], 1000)], 0)
            acc = (logits.argmax(-1) == ev_y).mean()
            nll = optax.softmax_cross_entropy_with_integer_labels(logits, ev_y).mean()
            return acc, nll

        params, opt_state = params0, optim.init(params0)
        key = jr.PRNGKey(42)                    # fixed across trials
        rng = np.random.default_rng(0)          # fixed batch order across trials
        out = []
        for e in range(epochs):
            perm = rng.permutation(N)
            for s in range(steps):
                b = perm[s * batch:(s + 1) * batch]
                key, sk = jr.split(key)
                params, opt_state, l = train_step(params, opt_state,
                                                  train_x[b], train_y[b], sk)
            if not bool(jnp.isfinite(l)):
                break
            acc, nll = eval_subset(params)
            if not bool(jnp.isfinite(nll)):
                break
            out.append((float(acc), float(nll)))
            if report is not None:
                report(e, (float(acc), float(nll)))
        return out

    return run


# ---------------------------------------------------------------- search spaces
def suggest_adamw(trial, sch):
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    wd = trial.suggest_float('wd', 1e-4, 2e-1, log=True)
    return optax.adamw(sch(lr), weight_decay=wd)


def suggest_ivon(trial, sch, N):
    lr = trial.suggest_float('lr', 3e-4, 3e-1, log=True)
    h0 = trial.suggest_categorical('hess_init', [1e-3, 1e-2, 1e-1, 0.35, 1.0])
    b1 = trial.suggest_categorical('b1', [0.8, 0.9, 0.95, 0.99])
    b2 = trial.suggest_categorical('b2', [0.95, 0.99, 0.999, 0.9999, 0.99999])
    essm = trial.suggest_categorical('ess_mult', [0.1, 1.0, 10.0, 100.0, 1000.0])
    clip = trial.suggest_categorical('clip_radius', [1e-3, 1e-2, 1e-1, 1.0, float('inf')])
    wd = trial.suggest_float('wd', 1e-7, 1e-2, log=True)
    return ivon(sch(lr), ess=N * essm, hess_init=h0, weight_decay=wd,
                clip_radius=clip, b1=b1, b2=b2)


def suggest_evon(trial, sch, N):
    lr = trial.suggest_float('lr', 3e-4, 3e-1, log=True)
    h0 = trial.suggest_categorical('hess_init', [1e-3, 1e-2, 1e-1, 0.35, 1.0])
    b1 = trial.suggest_categorical('b1', [0.8, 0.9, 0.95, 0.99])
    b2 = trial.suggest_categorical('b2', [0.95, 0.99, 0.999, 0.9999, 0.99999])
    b3 = trial.suggest_categorical('b3', [0.9, 0.95, 0.99])
    essm = trial.suggest_categorical('ess_mult', [0.1, 1.0, 10.0, 100.0, 1000.0])
    clip = trial.suggest_categorical('clip_radius', [1e-3, 1e-2, 1e-1, 1.0, float('inf')])
    wd = trial.suggest_float('wd', 1e-7, 1e-2, log=True)
    return evon(sch(lr), ess=N * essm, hess_init=h0, weight_decay=wd,
                clip_radius=clip, b1=b1, b2=b2, b3=b3,
                precond_every=10, max_precond_dim=10000, one_sided=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', default='/ptmp/cheins/data')
    ap.add_argument('--dataset', default='cifar100')
    ap.add_argument('--epochs', type=int, default=2)
    ap.add_argument('--batch', type=int, default=128)
    ap.add_argument('--eval-n', type=int, default=2000)
    ap.add_argument('--trials-adamw', type=int, default=10)
    ap.add_argument('--trials-ivon', type=int, default=40)
    ap.add_argument('--trials-evon', type=int, default=25)
    ap.add_argument('--objective', choices=['acc', 'nll'], default='acc',
                    help='acc: maximize epoch-K subset accuracy (optimization-speed '
                         'target). nll: minimize epoch-K subset NLL (calibration-aware '
                         'target; should steer zeta warm instead of cold).')
    ap.add_argument('--dim', type=int, default=192)
    ap.add_argument('--depth', type=int, default=6)
    ap.add_argument('--heads', type=int, default=3)
    ap.add_argument('--out', default='optuna_vit_results.json')
    args = ap.parse_args()

    print('jax', jax.__version__, '| devices', jax.devices(), flush=True)
    data = load_cifar(args.data_root, args.dataset)
    N = int(data[0].shape[0])
    model0 = build_model(jr.PRNGKey(0), data[4], dim=args.dim, depth=args.depth,
                         heads=args.heads)
    run_short = make_short_trainer(model0, data, epochs=args.epochs,
                                   batch=args.batch, eval_n=args.eval_n)
    warm = max(1, (N // args.batch) // 20)      # same 5%-of-epoch warmup as long runs

    def sch(peak):
        # warmup -> constant peak: mimics the EARLY phase of a long cosine run
        return optax.join_schedules(
            [optax.linear_schedule(0.0, peak, warm), optax.constant_schedule(peak)],
            [warm])

    results = {'meta': vars(args) | {'N': N, 'jax': jax.__version__}, 'studies': {}}

    def save():
        with open(args.out, 'w') as f:
            json.dump(results, f, indent=2, default=str)

    use_nll = args.objective == 'nll'
    direction = 'minimize' if use_nll else 'maximize'
    NAN_SCORE = 20.0 if use_nll else 0.0        # worse than any real trial

    def sweep(name, suggest, n_trials, estimator='sampling'):
        study = optuna.create_study(
            direction=direction, study_name=name,
            sampler=optuna.samplers.TPESampler(seed=0),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=0))

        def objective(trial):
            t0 = time.time()
            try:
                optim = suggest(trial)
            except Exception as e:
                raise optuna.TrialPruned(f'bad params: {e}')

            def report(epoch, acc_nll_pair):
                val = acc_nll_pair[1] if use_nll else acc_nll_pair[0]
                trial.report(val, epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            try:
                hist = run_short(optim, estimator, report=report)
            finally:
                jax.clear_caches()
            if not hist:                          # NaN'd in epoch 1
                return NAN_SCORE
            trial.set_user_attr('hist', hist)
            trial.set_user_attr('wall_s', round(time.time() - t0, 1))
            return hist[-1][1] if use_nll else hist[-1][0]

        study.optimize(objective, n_trials=n_trials,
                       callbacks=[lambda st, tr: _dump(st, name)])
        return study

    def _dump(study, name):
        results['studies'][name] = {
            'best_value': study.best_value if study.best_trial else None,
            'best_params': study.best_params if study.best_trial else None,
            'trials': [
                {'number': t.number, 'state': str(t.state), 'value': t.value,
                 'params': t.params, 'hist': t.user_attrs.get('hist'),
                 'wall_s': t.user_attrs.get('wall_s')}
                for t in study.trials],
        }
        save()

    fmt = (lambda v: f'nll={v:.3f}') if use_nll else (lambda v: f'{v*100:.2f}%')

    # 1) tuned-AdamW reference (the target to close on)
    s_adamw = sweep('adamw', lambda t: suggest_adamw(t, sch), args.trials_adamw)
    ref = s_adamw.best_value
    print(f'\n### AdamW reference (epoch-{args.epochs}, {args.objective}): '
          f'{fmt(ref)} with {s_adamw.best_params}', flush=True)

    # 2) IVON sweep
    s_ivon = sweep('ivon', lambda t: suggest_ivon(t, sch, N), args.trials_ivon)
    print(f'### IVON best: {fmt(s_ivon.best_value)} (ref {fmt(ref)}) '
          f'with {s_ivon.best_params}', flush=True)

    # 3) EVON sweep
    s_evon = sweep('evon', lambda t: suggest_evon(t, sch, N), args.trials_evon)
    print(f'### EVON best: {fmt(s_evon.best_value)} (ref {fmt(ref)}) '
          f'with {s_evon.best_params}', flush=True)

    print(f'\n=== SUMMARY (epoch-{args.epochs} subset, objective={args.objective}) ===')
    for name, st in (('adamw', s_adamw), ('ivon', s_ivon), ('evon', s_evon)):
        done = [t for t in st.trials if t.value is not None]
        top = sorted(done, key=lambda t: (t.value if use_nll else -t.value))[:5]
        print(f'--- {name} top-5 ---')
        for t in top:
            h = t.user_attrs.get('hist') or [(float('nan'),) * 2]
            print(f'  {fmt(t.value)}  acc={h[-1][0]*100:.1f}%  {t.params}')
    save()
    print('wrote', args.out, flush=True)


if __name__ == '__main__':
    main()
