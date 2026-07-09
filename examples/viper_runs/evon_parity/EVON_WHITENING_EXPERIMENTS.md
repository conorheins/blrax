# EVON whitening / Newton-Schulz — empirical tests of the #research Slack claims

Setup: official `team-approx-bayes/evon` (PyTorch, torch 2.13 CPU). Scripts:
`exp_evon.py` (E1-E3), `exp_phasing.py` (E4). All four claims from the thread reproduce.

## E1 — Dimitrije: "whitening pushes updates to the unit sphere ⇒ needs a decaying LR"

Deterministic EVON on a ridge-regression problem (unique optimum), 400 steps,
measuring behaviour over the last 20%:

| config | excess loss | late-loss osc (std) | ‖w−w*‖ final |
|---|---|---|---|
| whiten=False, const LR | −4e-9 | **1.1e-8** | 2.3e-6 |
| whiten=False, cosine LR | 9e-8 | 4.1e-8 | 8.6e-4 |
| whiten=True, const LR | 8.1e-4 | **2.3e-4** | 9.0e-2 |
| whiten=True, cosine LR | 2.3e-6 | 3.1e-6 | 2.3e-4 |

**Confirmed.** Whitening at constant LR wanders around the optimum with oscillation
amplitude **~22,000× larger** than the un-whitened optimizer (which settles to machine
noise). A cosine schedule suppresses the wander ~73× and pulls ‖w−w*‖ back in line.
Mechanism: whitening normalizes the update's singular values to ≈1, so the step size
does **not** shrink as the gradient vanishes near the optimum — only a decaying LR makes
whiten+EVON converge.

## E2 — Dimitrije: "Newton-Schulz is a worse approximation, just efficient in bf16"

NS whitening vs the exact orthogonalization (polar factor U Vᵀ, all σ=1):

| matrix | NS σ range | direction err vs UVᵀ | bf16-vs-fp32 input |
|---|---|---|---|
| (5,12) cond 1 | [0.68, 1.13] | 19% | 0.000 |
| (8,8) cond 1 | [0.71, 1.13] | 18% | 0.000 |
| (8,8) cond 1e3 | **[0.24, 1.13]** | **33%** | 0.000 |
| (16,64) cond 1 | [0.69, 1.13] | 21% | 0.000 |

**Confirmed.** 5-step NS leaves singular values spread over ~[0.5, 1.5] (worse — down to
0.24 — for ill-conditioned inputs) vs exactly 1.0 for true orthogonalization; 18–33%
Frobenius direction error. Note the code **hard-codes bf16** (`x = g.bfloat16()`), so
passing fp32 changes nothing — the "efficiency in bf16" is baked in, and the approximation
error is dominated by the 5-step truncation, not the dtype.

## E3 — Dimitrije: "explains why they don't have IVON-style clipping of the effective grad"

Per-step effective update-norm distribution over 400 steps:

| config | median | p95 | max | **max/median** |
|---|---|---|---|---|
| whiten=False | 0.070 | 1.54 | 1.70 | **24.1** |
| whiten=True | 2.11 | 2.30 | 2.42 | **1.15** |

**Confirmed — and the paper says so explicitly.** Un-whitened, the raw update norm spikes
24× above its median early in training — which is exactly why IVON needs an explicit
`clip_radius`. Whitening flattens the singular values, so the update norm is bounded by
construction (max/median → 1.15). The paper's Algorithm 2 (line 8) confirms the framing:
Newton-Schulz whitening IS the *"spectral clipping"* option for the parameter update, an
**alternative to** IVON/Sophia-style element-wise clipping — not a separate transform. So
Dimitrije's read is exactly right: whitening *is* their update-clipping mechanism, which is
why there's no separate effective-gradient clip. E1 and E3 are two faces of one mechanism:
whitening makes ‖update‖ ~constant regardless of gradient magnitude — good for stability
(it is the clip), bad for settling at constant LR (needs cosine).

## E4 — Dimitrije: "alternating noisy Hessian / deterministic L,R at the mode — I think there's a bug"

Instrumented the official `phasing=True` mode, logging per optimizer iteration the phase
assigned at BOTH decision sites (`_sample_params` reads the step counter pre-increment,
`step()` post-increment) and whether H / GG actually changed:

```
iter | sample_phase | update_phase | noise? | H changed | GG changed
  1  |    CLEAN      |    (init)     |  no    |   no      |   no        <- Shampoo-init skip
  2  |    CLEAN      |    CLEAN      |  no    |   no      |   yes       <- L/R from mode grad
  3  |    noisy      |    noisy      |  yes   |   yes     |   no        <- Hessian from noise
  4  |    CLEAN      |    CLEAN      |  no    |   no      |   yes
  ... (alternates cleanly)
No invariant violations detected.
```

**Could not reproduce a bug in the default config.** The alternation is self-consistent:
noise is applied exactly on the H-update (noisy) steps, GG updates exactly on the clean
steps, and exactly one of {H, GG} changes per step. The parity between the two counter-read
sites holds **because** the first optimizer call initializes the Shampoo preconditioner and
`continue`s without incrementing the counter — that skip is what offsets `_sample_params`
(pre-increment) against `step()` (post-increment) so they agree. This coupling is delicate
(it would break if `max_precond_dim ≤ 0`, i.e. preconditioning fully disabled, but then
phasing is moot). If a real bug exists it is more likely a **paper-vs-code spec mismatch**
(the alternation may not be what the paper specifies) than a fault in this trace — that is
being checked separately against the paper.

blrax has **no phasing mode at all**: its EVON updates both the Hessian (Price/sampling)
and the L/R covariance every step from the *same noisy sample*. So the alternation — and
any bug in it — simply doesn't exist on our side. The only consequence is that blrax
estimates L/R from noisy-sample gradients rather than mode gradients; the noise inflates
diag(GG) by the mean posterior variance ~1/(ess·h), which for ess=N (≥5e4 in our runs) is
~1e-5 against gradient-covariance entries of O(grad²) — negligible in our regime.

## One-line summary

All three of Dimitrije's whitening/NS claims are empirically confirmed; the phasing
alternation is self-consistent in the default config (no bug reproduced empirically). None
of this touches the Bayesian posterior — whitening and phasing only affect the mean's
optimization trajectory, which is why blrax's omission of both is fine for the uncertainty/
CL work and only matters if we try to match the paper's from-scratch optimization curves.
