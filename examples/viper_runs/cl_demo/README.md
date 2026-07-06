# Continual-learning demo: Bayesian finetuning (IVON/EVON) vs AdamW

**Setup.** Split CIFAR-100, 10 tasks x 10 classes, class-incremental (single
shared 100-way head, no task id at test, CE masked to classes seen so far),
replay-free. Head = linear probe on frozen DINOv2 ViT-S/14 CLS features
(equimo `dinov2_vits14(pretrained=True, dynamic_img_size=True)`, 384-d,
CIFAR upscaled to 224 bicubic + ImageNet norm; `extract_dino_features.py`).
20 epochs/task, batch 128, 3 seeds (seeded class order).

**Bayes-CL mechanism (`ivon-cl`/`evon-cl`)** — VCL recursion with unmodified
blrax: carry the optimizer state across tasks AND add a quadratic penalty
`(1/(2*zeta_t)) * sum_i Lam_prev_i (theta_i - mu_prev_i)^2` to the per-sample
mean loss, where `Lam_prev = 1/get_scale(opt_state)^2 = zeta*(h+wd)` is the
previous task's posterior precision (snapshot BEFORE the ess swap at each
boundary; wd=1e-7 so the built-in zero-prior never double counts; boundary
surgery via NamedTuple._replace, never optim.init; EVON uses the diagonal
Kronecker-marginal precision and never resets count).

**Results (results_cl_final.json, mean +/- std over 3 seeds):**

| method     | final ACC     | forgetting | union NLL |
|------------|---------------|-----------|-----------|
| AdamW naive| 29.3 +/- 0.6  | 75.6      | 4.08      |
| AdamW+EWC  | 58.5 +/- 1.7  | 42.1      | 2.03      |
| IVON naive | 54.3 +/- 1.9  | 47.5      | 1.88      |
| IVON-CL    | 72.5 +/- 0.8  | 21.8      | 1.29      |
| EVON-CL    | 72.6 +/- 0.9  | 21.6      | 1.29      |
| joint ceiling (IVON, all classes at once) | 87.6 | - | 0.42 |

Literature anchors (ViT-B/16 IN21k, 10x10 CIL): FT-seq 33.6 / EWC 47.0 /
L2P 83.8 / NCM 83.4 / joint probe 87.9. Our AdamW naive (29.3) and joint
(87.6, DINOv2 S/14) match; IVON-CL recovers 83% of the joint ceiling
replay-free, +43 pts over naive AdamW, +14 over EWC with exact Fisher.

**Tuning notes.** ivon-cl/evon-cl best: lr 0.1, b2=0.99999, tau=10 (zeta=
10*N_task), h0 0.5, p0 1.0, clip off. b2=0.99 (fast Hessian EMA) is
catastrophic (~56%, NLL explodes) — the 1-MC gradient-noise Hessian estimate
is too noisy at 780 steps/task; the near-isotropic slow-EMA Lam beats noisy
adapted Lam. tau is a no-op for retention (algebra: penalty weight is
zeta-invariant). lr 0.1 trades a little plasticity (LA 97->92) for less
forgetting. Config surface is flat 72-74%: the plateau of pure replay-free
weight regularization; prototype/replay tricks (NCM/RanPAC) needed to go
higher.

Sweeps: `run_cl_sweep.sbatch` (b2 x lr x tau grid), `run_cl_sweep2.sbatch`
(lr/clip/epochs polish). Full per-task R matrices in the JSONs.
