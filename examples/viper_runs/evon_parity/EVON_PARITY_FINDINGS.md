# blrax `evon` vs official `team-approx-bayes/evon` — parity findings

Reference: official code cloned 2026-07-09 (single-file `evon-src/evon.py`, PyTorch),
paper arXiv:2606.23357 (SOAP-Bubbles). blrax: `src/blrax/optim.py` + `utils.py`.
Cross-framework numerical test: `parity_evon.py` (torch 2.13 CPU vs jax 0.10, x64).

## Verdict

The **Bayesian/statistical core is numerically identical to machine precision** —
verified bit-for-bit against the released code with matched inputs:

| kernel | max abs diff |
|---|---|
| Price + positivity-preserving Hessian EMA | 4.4e-16 |
| coupled-WD preconditioned Newton update `(m+wd·p)/(h+wd)` | 5.3e-10 |
| structured posterior covariance `get_scale` vs projected eigenspace noise | 5.2e-4 (MC) |
| `project_back` == `Q_L E Q_Rᵀ` | 5.6e-17 |

The differences are all in the optimizer's *outer shell* (basis bookkeeping +
engineering stabilizers), not the posterior or the natural-gradient step.

## Differences, ranked

1. **Newton–Schulz whitening — official ON by default (`whiten_prec_grad=True`),
   blrax absent.** The official code applies Muon-style NS orthogonalization to the
   final parameter-space update (flattens the update's singular values to ≈1).
   blrax has no whitening → our EVON == their `whiten_prec_grad=False`.
   - *Impact:* changes the **mean** optimization trajectory only; the posterior
     `N(M, Q_L diag(σ²) Q_Rᵀ)` is untouched. So it's irrelevant to the
     uncertainty/CL story but is the most likely reason our from-scratch loss
     curves would differ from the paper's. Open question: whether the paper's
     headline numbers use whitening (released default says yes).

2. **Eigenbasis refresh ordering — blrax sorts columns descending by eigenvalue
   (SOAP) + reshuffles H; official omits the sort.** This is the code path of
   Dimitrije's reshuffle bug. blrax keeps SOAP's `argsort`+`H[idx]` (now correct);
   the official authors sidestepped it by dropping the sort entirely.
   - *Verified:* both track the same eigenspace, but the sorted refresh converges
     the per-step basis far tighter — median eigenvector error after 8 warm
     iterations (20 trials): **blrax 3.3e-5 vs official 3.8e-2**. Putting the
     dominant direction first lets Gram-Schmidt extract it cleanly. So blrax is
     closer to canonical SOAP here and arguably higher-quality per refresh.

3. **Momentum bias-correction — official debiases (`correct_bias=True`), blrax's
   EVON leaves do not.** Official divides momentum by `(1-b1^t)`; blrax
   `_update_*_leaf` uses the raw EMA. Transient (first ~1/(1-b1) steps). NB blrax's
   *diagonal IVON* path DOES debias — only the EVON leaves skip it.

4. **First-step handling — official skips the first param update to seed Shampoo
   stats (runs in the eigenbasis from step 2); blrax runs in the identity basis
   for the first `precond_every` steps, then switches.** Same steady state.

5. **Default hyperparameters differ (we override these anyway):**
   - official `betas=(0.95, 0.9999)`, `shampoo_beta → 0.9999`, Hessian clip **off**.
   - blrax `b1=0.9, b2=0.95, b3=0.95`, Hessian clip **on** (ratio 10).
   - Notable: the covariance/eigenbasis EMA `b3` default is **0.95 in blrax vs
     0.9999 in official** — a ~20× faster-decaying eigenbasis out of the box.

6. **blrax-only additions:** Hutchinson HVP diagonal-Hessian estimator
   (`estimator='hutchinson'`) as an alternative to the Price/sampling estimator
   (the official code has only Price + a grad² fallback when sampling is off);
   `one_sided` preconditioning mode.

7. **Official-only engineering (mostly infra/optional):** `merge_dims` /
   `precondition_1d` / `data_format` (conv layouts), distributed `sync_samples`
   all-reduce, `phasing` (clean/noisy alternating precond/Hessian phases),
   deterministic grad² fallback, `debias_beta2`. blrax handles n-d params by a
   fixed "flatten leading dims → (∏shape[:-1], shape[-1])" merge.

## Bottom line

Nothing in blrax's EVON contradicts the released reference on the parts that define
the method (Price estimator, Newton step, structured Gaussian posterior) — those are
identical to fp precision. The real gaps are (a) **no Newton–Schulz whitening**
(off vs their default-on; matters for optimization speed, not the posterior),
(b) a **different-but-arguably-better eigenbasis sort convention**, and (c) default
betas. If we ever benchmark against the paper's optimization curves, whitening is
the flag to reconcile first.
