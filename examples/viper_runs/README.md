# viper_runs — EVON/IVON benchmarking on MPCDF clusters (Jun–Jul 2026)

Results + scripts from the blrax EVON/IVON/AdamW comparisons run on Raven
(A100/CUDA) and Viper-GPU (MI300A/ROCm). All runs use the repo's blrax at or
after commit 91be154 (eigenbasis H-reshuffle fix). CIFAR staged as npz from the
HF parquet mirrors (cs.toronto.edu was throttled).

## results/
- results_evon_vs_ivon_v3.json — Raven A100. MNIST MLP 512x3, 15 ep. Notebook
  recipe (lr 1e-2, hess_init .35, clip .1). adamw/ivon/evon/h-evon: dead heat,
  Bayes wins NLL/ECE.
- results_vit_cifar100.json — Viper. Hand-rolled ViT 0.8M, CIFAR-100, 20 ep,
  no aug. EVON slightly > IVON; AdamW ECE blows up.
- results_vit_equimo.json — Viper. equimo ViT 2.7M, 20 ep, no aug. b2 recipe
  test: EVON b2=0.95 (paper finetuning value) DIVERGES from scratch at zeta=N;
  b2=0.9999 stable. Paper's b2=0.95 is coupled to their huge zeta (1e7-1e10).
- results_vit_equimo_aug.json — Viper. 60 ep + crop/flip aug, b2=0.9999.
  IVON 47.5 / EVON 46.9 / AdamW 51.0; Bayes ECE ~0.03 vs AdamW 0.31.
  EVON==IVON (gap did NOT widen with aug/epochs).
- optuna_vit_results.json — Viper. 2-epoch Optuna sweeps (objective = early
  ACC vs tuned-AdamW ref; Dimitrije's protocol). IVON best: lr .0067,
  hess_init .01, b1 .8, b2 .99999, zeta=1000N. EVON best: lr .11, hess_init
  .35, b1 .9, b2 .99999, b3 .9, zeta=10N.
- results_vit_tuned.json — Viper. 60-ep confirmation with tuned configs.
  IVON* 54.9% Pareto-dominates AdamW* 50.9%; EVON* 53.0% (better NLL 2.95,
  BMA still helps). BUT acc-objective went cold (zeta up) -> ECE ~0.3 for all.
  zeta = the acc/calibration dial; hand configs (zeta=N) are the warm end
  (47%, NLL 2.03, ECE 0.03).
- smoke*.json — 1-2 epoch smoke tests.

## scripts/
- evon_vs_ivon_mnist.py — Raven MNIST run (uses repo data_loaders/training).
- vit_cifar_evon.py — hand-rolled ViT CIFAR run (ROCm-safe harness).
- vit_cifar_equimo.py — equimo VisionTransformer run + aug (main harness).
- tune_vit_optuna.py — short-run Optuna sweeps (--objective acc|nll).
- confirm_tuned_vit.py — long run reading best_params from sweep JSON.
- run_*.sbatch — SLURM wrappers (Viper apu/apudev; two-stage Lmod load).

## Gotchas discovered
- ROCm XLA segfaults on the big fused graph (scan-over-epochs + full-dataset
  in-jit gather + 10k eval). Fix: host shuffle, python step loop with small
  jitted train_step, chunked eval. Same code fused fine on CUDA.
- Viper-CPU and Viper-GPU do NOT share $HOME or /ptmp (AGENTS.md was wrong).
- equimo ViT path needs only banax/einops/loguru extra deps (--no-deps safe
  on top of the MPCDF jax/0.8.2 ROCm module).

## Addendum: NLL-objective iteration (Jul 5)
- optuna_vit_nll.json — same sweep with epoch-2 NLL objective. IVON's corner
  flips to the IVON-paper transformer trick (clip 1e-3 + lr 0.29, zeta=100N);
  EVON's corner unchanged (zeta=10N, b3=0.9). Epoch-2 NLL does NOT push zeta
  warm (overconfidence hasn't developed by epoch 2).
- results_vit_nlltuned.json — 60-ep confirmation: the NLL-tuned IVON is WORSE
  on everything at 60 ep (49.4% / NLL 3.91 / ECE 0.35) than the acc-tuned one
  (54.9 / 3.33 / 0.31): 2-epoch proxies mislead when rankings flip over
  training. NLL-tuned EVON replicates the acc-tuned EVON (~53%, best BMA NLL
  3.01). Conclusion: short proxies tune optimization speed only; pick the
  calibration point by constraining zeta (or early-stop on val NLL).

## Addendum 2: zeta=N sweep per Dimitrije's recipe (Jul 5, evening)
- optuna_vit_zn.json — ess=N FIXED (his point: free zeta corrupts posterior
  scale), cosine proxy (4 ep), spaces around his GPT-2 recipe (lr 0.1, clip
  1e-2 seeded). IVON corner: lr .165, clip .003, h0 .1, b2 .99999. EVON corner:
  lr .178, clip .03, h0 .35, b2 .999, b3 .9 — EVON out-swept IVON at ep 4
  (31.0 vs 30.1) for the first time.
- results_vit_zntuned.json — 60-ep confirmation: **IVON@zeta=N is the overall
  winner across all runs** — 55.7% acc (best), BMA NLL 1.95 (best), BMA ECE
  0.13; strictly dominates tuned AdamW (50.0/3.13/0.30) and beats the cold
  zeta=1000N config on acc AND NLL. BMA is functional again (NLL 2.27->1.95,
  ECE 0.22->0.13). EVON's ep-4 winner (b2=.999, lr .178) DIVERGED over 60 ep
  (8%): second mistransfer; EVON's stable corner remains zeta=10N/lr~0.1/
  b2=.9999 (53%). Takeaway: Dimitrije's high-lr+tight-clip@zeta=N recipe is
  the right regime for IVON; EVON needs slower b2 for long horizons.

## Addendum 3: stabilized EVON at zeta=N (Jul 6)
- optuna_vit_zn_evon2.json — EVON-only re-sweep at ess=N with b2 constrained
  to {0.9999, 0.99999} (0.999 won the short proxy then diverged long-horizon)
  and a 6-epoch cosine proxy; seeds: Dimitrije recipe, IVON-winner transplant,
  zeta=10N corner. Winner: lr 0.185, clip 0.003, b2 0.99999, h0 0.35, b3 0.9.
- results_vit_zn_evon2.json — 60-ep confirmation: EVON@zeta=N 54.4% / NLL 2.63
  / ECE 0.26 (BMA 54.5 / 2.36 / 0.21). Divergence fully fixed (8% -> 54.4%),
  beats tuned AdamW (+4.4 acc, much better NLL), BMA functional. Still behind
  IVON@zeta=N on every metric (55.7 / BMA 1.95 / 0.13) at ~5x wall-clock.
  CAMPAIGN VERDICT: Dimitrije's high-lr+tight-clip@ess=N recipe makes both
  Bayesian methods dominate tuned AdamW from scratch; IVON > EVON on every
  metric for from-scratch ViT at this scale (EVON's case stays finetuning/LM).
