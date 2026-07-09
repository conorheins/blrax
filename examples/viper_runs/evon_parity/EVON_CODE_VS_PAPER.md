# EVON released code vs the SOAP-Bubbles paper (Algorithm 2) + phasing bug verdict

Produced by a 6-agent workflow: 1 agent diffed the ar5iv paper text against the
released code; 3 adversarial agents independently traced the phasing logic; 1 judge
reconciled; 1 synthesized. Cross-validated against my empirical instrumentation (exp_phasing.py).

---

## Paper-vs-code spec diff

I have everything I need. Let me note the key point about `Ĥ ← clip(E⊙G°/V)` and reconcile it with the code's `f = avg_nxg * (h + wd) * ess`, since `1/V = ζ(H+δ)`, so `E⊙G°/V = ζ(H+δ)·E⊙G°` = `ess·(h+wd)·(noise·grad)`. That matches the code. Good — the estimator math agrees.

Now let me verify a few specific code behaviors against the paper before writing the diff: the clip target (paper clips U in line 8 spectrally; code clips f element-wise AND whitens U), the phasing, and the bias correction. I have all I need. Let me write the report.

## Paper spec

**Paper:** "SOAP-Bubbles: Structured Weight Uncertainty for Neural Networks" (arXiv:2606.23357), Minut, Daheim, Miani, Khan, Lin, Möllenhoff (2026). Introduces EVON / Eigenspace-VON: IVON-style variational learning run inside a SOAP/Shampoo eigenbasis `(Q_L, Q_R)`.

**Algorithm 2 (EVON), verbatim from ar5iv full text:**

```
1  V_{i,j}  ← 1/(ζ(H_{i,j}+δ))                         # posterior variance in eigenspace
2  Θ        ← M + Q_L E Q_R^⊤,   E_{i,j} ~ N(0, V_{i,j})   # sample weights
3  G ← ∇̂ℓ(Θ),   G° ← Q_L^⊤ G Q_R                       # grad, projected into eigenbasis
4  Ĥ ← clip(E ⊙ G° / V)                                # Price/Stein Hessian estimate
5  Ḡ ← β₁Ḡ + (1−β₁)G°                                  # momentum (eigenspace)
6  H ← β₂H + (1−β₂)Ĥ + ½(1−β₂)²(Ĥ−H)^⊙2/(H+δ)          # Hessian EMA + positivity correction
7  U ← (Ḡ + δ Q_L^⊤ M Q_R)/(H+δ)                       # preconditioned update (coupled WD)
8  M ← M − α·clip(Q_L U Q_R^⊤)                          # mean update in param space
9  L ← β₃L + (1−β₃)GG^⊤,  R ← β₃R + (1−β₃)G^⊤G          # Shampoo statistics
10 Every T steps: Q_L, Q_R ← Eig(L), Eig(R)             # refresh eigenbasis (project momentum)
```

Key points confirmed verbatim from the paper text:
- **Hessian estimator**: `Ĥ ← clip(E ⊙ G°/V)`. Since `1/V = ζ(H+δ)`, this equals `ζ·(H+δ)·(E⊙G°)`. Positivity correction term is `+½(1−β₂)²(Ĥ−H)^⊙2/(H+δ)` (line 6).
- **ζ (zeta)** is the ELBO's expected-loss weight (Eq. 1: `ζ·E_q[ℓ(θ)] + KL`). Posterior variance `V = 1/(ζ(H+δ))` is inversely proportional to ζ. This plays the role of the code's `ess`.
- **δ (delta)** is weight-decay / prior precision; appears in the update denominator `(H+δ)` and as coupled WD in the numerator `+δ Q_L^⊤ M Q_R` (line 7).
- **Clipping** (Section 3.2): *"Line 8 uses a clipping of the preconditioned gradient. EVON has the option to either use element-wise clipping as in IVON or the Sophia optimizer, or to use spectral-clipping."* Spectral clipping *"is implemented using a quintic Newton-Schulz iteration, as in Muon."* So Newton-Schulz is **one optional choice for the clip() on line 8** — the final param update `U` — and it is an **alternative to** element-wise clipping, not applied in addition to it. Clip ratio γ defaults to **10**.
- There is **no clean/noisy phasing / alternation**. Sampling (lines 1–2) and Hessian updates (lines 3–6) run **every iteration**; only the eigenbasis refresh is periodic (every T steps, line 10), independent of noise scheduling.
- **No consolidated default hyperparameter table**; γ=10 and 32 MC samples for model averaging are the only firm numbers. β₁/β₂/β₃, ζ, H₀, T, α, WD, and bias-correction are not fixed by the paper (per-experiment, appendix E.3/E.4). No mention of Adam-style bias correction anywhere.
- Eigenbasis refresh: momentum is re-projected into the new space, but *"both methods do not update H into the new space."* The paper says **nothing** about flipping eigenvalue sign/order or re-sorting H when the basis rotates.

## Code

File: `.../evon_official/evon-src/evon.py`. `EVON(Optimizer)` implements the algorithm above with these defaults and behaviors:

- `__init__` defaults (lines 40–59): `lr=3e-3`, `betas=(0.95, 0.9999)`, `shampoo_beta=-1` (→ falls back to `betas[1]=0.9999`), `eps=1e-10`, `weight_decay=1e-6`, `precondition_frequency=10`, `correct_bias=True`, `mc_samples=1`, `phasing=False`, `price_clip_ratio=None`, **`whiten_prec_grad=True`**, `debias_beta2=False`. `ess` and `hess_init` are required.
- Hessian estimator `_price_hess_update` (lines 742–767): `f = avg_nxg * (h + wd) * ess`; optional element-wise clamp to `±clip_ratio*(h+eps)`; then `h ← β₂h + (1−β₂)f + ½(1−β₂)²(h−f)²/(h+wd+eps)`. Matches paper lines 4 & 6 (with `ess`↔ζ, `wd`↔δ).
- Momentum `exp_avg.mul_(beta1).add_(grad_proj, 1−beta1)` (line 286) — matches line 5.
- Update (lines 313–325): `preconditioner = h_mom + wd + eps`; `precond_grad = (exp_avg/debias1 + wd·p_proj)/preconditioner` — matches line 7.
- **Whitening** (lines 335–337): after projecting `U` back to param space, if `whiten_prec_grad` (default True) it calls `_whiten(update)` = Newton-Schulz orthogonalization (`_zeropower_via_newtonschulz`). This is applied **in addition to** the optional element-wise `price_clip_ratio`, and it is on by default.
- **Phasing** (lines 275–284, 290, 336–343, 559–573): a clean/noisy alternation gated on `phasing` (default False). On clean (odd) steps: noise=0, preconditioner updates, Hessian does NOT; on noisy (even) steps: noise added, Hessian updates, preconditioner does NOT (except every `precondition_frequency`).
- Bias correction (lines 307–310): `debias1 = 1−β₁^step` always (when `correct_bias`); `debias2 = 1−β₂^step` only if `debias_beta2=True` (default False); `step_size = lr/debias2`.

## Divergences (ranked)

**1. Whitening (Newton-Schulz) is ON by default in code; the paper treats it as an OPTIONAL alternative to element-wise clipping — and the code applies it as an unconditional extra transform, not as "the clip()." — MATERIAL**
- Paper: line 8 `clip(...)` is *either* element-wise clipping *or* spectral (Newton-Schulz) clipping — mutually exclusive choices, and clipping is a bounded operation on `U`.
- Code: `whiten_prec_grad=True` default (line 58). `_whiten` (lines 831–836) fully orthogonalizes the update via `_zeropower_via_newtonschulz`, which rescales singular values to ≈Uniform(0.5,1.5) — it does not "clip/bound" the existing update, it **replaces its spectrum** (Muon-style). It runs regardless of `price_clip_ratio`, so you can get both element-wise clip of the Hessian AND full whitening of the update. Location: `evon.py:335-337`, default at `:58`.
- Impact: **Large.** Default behavior orthogonalizes every 2-D update every step. This changes the effective update geometry and magnitude versus the paper's default (element-wise clip), and stacks with the Hessian preconditioner rather than substituting for a clip. A faithful default would be `whiten_prec_grad=False` (element-wise clip as the primary path), with whitening as an opt-in for the "spectral clipping" variant.

**2. Clip target/semantics differ: paper clips the parameter update `U` (line 8); code clips the raw Hessian estimate `f` (line 4-analog), and its "spectral clip" is whitening of `U`. — MATERIAL**
- Paper: `Ĥ ← clip(E⊙G°/V)` (line 4) is the Hessian estimate; the paper's *element-wise vs spectral clipping* discussion is about **line 8's `clip(Q_L U Q_R^⊤)`**, i.e. the update.
- Code: element-wise clipping (`price_clip_ratio`) is applied to the **Hessian residual `f`** inside `_price_hess_update` (lines 762–765), matching line 4. But the paper's line-8 element-wise clip of `U` is **absent** — there is no bounded element-wise clip on `update`; only whitening. So the two "clip" sites in the paper (line 4 Hessian, line 8 update) are collapsed/relocated in code: line-4 clip exists, line-8 element-wise clip does not, and line-8 spectral clip is implemented as unconditional whitening.
- Impact: **Material.** With defaults (`price_clip_ratio=None`, `whiten_prec_grad=True`), the code applies *no* clip to the Hessian and *whitening* to the update — neither of the paper's line-8 element-wise-clip defaults. γ=10 (paper's default clip ratio) maps to `price_clip_ratio` in code but defaults to `None` (off).

**3. Phasing / clean-noisy alternation exists in code but is NOT in the paper. — MATERIAL when enabled, immaterial by default**
- Paper: no phasing/alternation; sampling and Hessian updates happen every step, preconditioner refresh every T steps independently (confirmed absent in full-text read).
- Code: `phasing` flag (default `False`) implements clean(odd)/noisy(even) alternation — clean steps zero the noise, update the preconditioner, and skip the Hessian; noisy steps add noise, update the Hessian, and skip the preconditioner. Locations: `evon.py:275-284, 290, 336-343, 559-573`.
- Impact: Off by default so it does not affect the default run, but it is an **unpublished algorithmic variant** not described in the paper. If a user enables it, results diverge from the paper spec entirely.

**4. `ess`/ζ default and semantics: code requires `ess` (no default) and multiplies `f` by `ess`; paper's ζ has no fixed value. — MINOR (consistent, but unpinned)**
- The mapping is correct: `f = avg_nxg*(h+wd)*ess` ⇔ `Ĥ = E⊙G°/V` with `1/V=ζ(h+δ)`. But the paper gives no default ζ, and code makes `ess` a required arg — so no numeric divergence, just no canonical value to check against. Not a bug.

**5. Bias correction: code has Adam-style `1−β^step` debiasing (on by default for β₁); paper's Algorithm 2 shows none. — MINOR/MATERIAL**
- Paper: Algorithm 2 has plain EMAs with no `1−β^t` bias correction terms; bias correction is not mentioned.
- Code: `correct_bias=True` default → `debias1 = 1−β₁^step` divides the momentum (`exp_avg/debias1`, line 324); `debias2` only if `debias_beta2=True` (default off). Location `:307-310, :324`.
- Impact: Material in the first ~1/(1−β₁) steps (with β₁=0.95, first ~20 steps), then negligible. It is an addition beyond the paper's written update but a standard, benign one. `debias2` off by default matches the paper (no β₂ debias).

**6. Default betas differ from typical IVON/SOAP and are not pinned by the paper. — MINOR (cannot confirm)**
- Code defaults `betas=(0.95, 0.9999)`, `shampoo_beta=-1`→0.9999. Paper does not state β₁/β₂/β₃ defaults, so these cannot be confirmed or refuted against the paper — flag as **inferred/unverifiable**, not a confirmed divergence.

**7. Eigenbasis refresh does not re-sort/flip H when the basis rotates. — MINOR, matches paper text but note the repo already patched this elsewhere**
- Paper explicitly says H is **not** updated into the new space on eigenbasis refresh — matching the code (`_update_preconditioner` re-projects `exp_avg` but not `h_mom`). No divergence in `evon.py`. (Note: repo commit `91be154` "re-sort diagonal Hessian to follow eigenbasis column permutation" suggests a related fix lives outside this file; not present/needed here per the paper.)

**8. First-step skip and `max_precond_dim>0` early-continue. — COSMETIC**
- Code skips the parameter update on the first encounter of each param when `max_precond_dim>0` (lines 256–258) to initialize the eigenbasis. Not described in the paper's algorithm box but a standard warm-up detail; immaterial beyond one step.

## Answer: does the code match the paper?

**Partially — the core math matches, but the default configuration does not.** The eigenspace momentum, the Price/Stein Hessian estimator `f = ess·(h+wd)·(noise·grad)` = `E⊙G°/V`, the positivity-correction EMA `+½(1−β₂)²(Ĥ−H)²/(H+δ)`, the coupled-weight-decay update `(Ḡ+δ·Q_LᵀMQ_R)/(H+δ)`, and the "H is not re-projected on eigenbasis refresh" behavior all faithfully implement Algorithm 2.

The divergences are in the **clipping/whitening path and the defaults**, and they are material:
- The paper's line-8 `clip(U)` is *either* element-wise *or* spectral (Newton-Schulz), as a **bounded clip alternative**. The code instead ships `whiten_prec_grad=True` by default, applying **full Newton-Schulz orthogonalization** to every 2-D update every step — an unconditional Muon-style spectral transform, not a clip, and stackable with the (defaulted-off) element-wise Hessian clip. To reproduce the paper's default you must set `whiten_prec_grad=False` and set `price_clip_ratio` to the paper's γ=10.
- The **phasing** clean/noisy alternation in the code is **not in the paper at all** (off by default, but an unpublished variant if switched on).
- Adam-style β₁ **bias correction** is added beyond the written algorithm (benign, early-step only).

So: correct estimator and update equations, but the released default (`whiten_prec_grad=True`, `price_clip_ratio=None`, `phasing=False`) does **not** correspond to the paper's default "element-wise clipping" EVON. Anyone running `EVON(...)` out of the box gets the whitened variant, which is at best the paper's optional spectral-clipping mode — and even then applied as orthogonalization rather than as a clip.

Code file audited: `/private/tmp/claude-501/-Users-conorheins-Documents-evora-ai-research-blrax--claude-worktrees-crazy-torvalds-4da37c/5daca920-f6b2-449e-8b28-c0134a3e4fc2/scratchpad/evon_official/evon-src/evon.py`

---

## Phasing bug verdict

I've now verified the decisive code paths. All three reviewers' traces are consistent with the actual code. Let me confirm the two key facts:

1. **Line 273** `state["step"] += 1` is AFTER the **line 257-258** first-encounter `continue` (increment is skipped on first encounter when `max_precond_dim > 0`). Confirmed.
2. **Sample site (line 563)**: `is_clean_phase = (current_step % 2 == 0) and phasing`, read BEFORE increment. **Step site (lines 281-283)**: `is_clean_phase = (current_step % 2 != 0 or not sampling_enabled) and phasing`, read AFTER increment. Confirmed the polarity flip (`==0` vs `!=0`) cancels the `+1` offset.

All three reviewers agree (no bug on the parity lens), all high-confidence, and their traces are mutually consistent and match the code. The verdict holds.

## Verdict

**No-bug** on the specific lens investigated (clean/noisy phase-parity consistency between `_sample_params` and `step`), and **moot / not-applicable** for our blrax usage. All three reviewers independently reached `bug_found: false` at high confidence, and I confirmed their traces against the actual source. There is no parity mismatch: no iteration ever applies posterior noise while treating that same iteration as "clean" (or vice-versa).

## The precise mechanism

The suspected bug is that the two phase-detection sites use *different code*:

- **Sample time** (`_sample_params`, line 563): `is_clean_phase = (current_step % 2 == 0) and phasing`, reading `state["step"]` **before** it is incremented.
- **Update time** (`step`, lines 281-283): `is_clean_phase = (current_step % 2 != 0 or not sampling_enabled) and phasing`, reading `state["step"]` **after** the line-273 increment.

These look contradictory (`==0` vs `!=0`), which is what triggered the bug suspicion. But they are algebraically identical because of an exactly-cancelling off-by-one:

- Within one iteration, `_sample_params` runs first and reads `S`; `step()` increments to `S+1` and reads that.
- `S % 2 == 0`  ⟺  `(S+1) % 2 != 0`. The `+1` offset from reading pre- vs. post-increment is precisely cancelled by the flipped `==`/`!=` operator.

The one wrinkle is the **first-encounter `continue`** (lines 257-258): on the very first `step()` for each parameter, `Q` is absent, so it inits + updates the preconditioner and `continue`s **before** line 273, so `state["step"]` is *not* incremented and stays `0`. This creates a permanent `+1` offset between the sample-read `S` and the step-read `S+1` for all subsequent iterations — which is exactly what makes the `==0`/`!=0` polarity flip line up.

Decisive trace (phasing=True, sampling on, mc=1, large `max_precond_dim`), agreed by all three reviewers and confirmed against code:

| Iter | sample reads S → phase | step reads → phase | noise? | Hessian? | GG (Shampoo)? |
|---|---|---|---|---|---|
| 1 | 0 → CLEAN | (continue, S stays 0) | no | no | yes (init, unconditional) |
| 2 | 0 → CLEAN | 1 → CLEAN | no | no | yes |
| 3 | 1 → NOISY | 2 → NOISY | yes | yes | no |
| 4 | 2 → CLEAN | 3 → CLEAN | no | no | yes |
| 5 | 3 → NOISY | 4 → NOISY | yes | yes | no |

Every iteration: the noise-applied flag matches the step-site phase, and exactly one of {Hessian update, Shampoo/GG update} fires. Iter 1-2 are a benign "double-clean" warm-up (first Price Hessian update deferred to iter 3), which matches the "leave h_mom unchanged" comment. No noisy sample is discarded; no clean sample is consumed as noisy.

## When it bites

- **Default `phasing=False` path:** Not affected. Both `is_clean_phase` expressions are `... and phasing`, so with `phasing=False` both evaluate to `False` at every site. No phase gating happens at all; the parity question is vacuous.
- **The paper's results (phasing=True):** Not bitten by *this* parity concern — the phase assignment is self-consistent. (Two *separate, unverified* concerns were flagged in passing and are not parity bugs: reviewer 1 notes the extra `or step % precondition_frequency == 0` clause at line 343 lets the Shampoo/GG accumulator also update on noisy steps, a possible paper-vs-code mismatch; and line 282's `or not self._sampling_enabled` term has no analog in `_sample_params`, but it's moot because `sampled_params()` skips `_sample_params` entirely when sampling is off. Neither is the phasing bug that was hunted.)
- **Our blrax usage:** Completely unaffected — blrax has no phasing at all, so this whole code region is inert for us. This is not a blocker for anything on our side.

## What to tell Dimitrije

The suspected EVON phasing bug is **not a bug**: the `%2==0` (pre-increment, in `_sample_params`) vs `%2!=0` (post-increment, in `step`) mismatch is intentional and self-cancelling — `S even ⟺ S+1 odd` — so the clean/noisy assignment is consistent at every iteration, and the first-step `continue` (which skips the `step` increment) is exactly what creates the `+1` offset that makes the two flipped comparisons agree. It does not affect the paper's results and is entirely irrelevant to our blrax port, which uses no phasing. Two unrelated things worth mentioning to the author only as minor asides, not blockers: line 343's `or step % precondition_frequency == 0` also refreshes the Shampoo accumulator on noisy steps (possible spec-vs-code discrepancy), and the two phase predicates are duplicated rather than sharing a helper (fragile if either is ever edited).

Source: `/private/tmp/claude-501/-Users-conorheins-Documents-evora-ai-research-blrax--claude-worktrees-crazy-torvalds-4da37c/5daca920-f6b2-449e-8b28-c0134a3e4fc2/scratchpad/evon_official/evon-src/evon.py` — decisive lines: 257-258 (first-step `continue`), 273 (increment), 281-283 (step-site predicate), 342-350 (GG update gate), 562-573 (sample-site predicate).

---

## Synthesis

I'll synthesize the briefing directly from the two inputs provided. No further investigation is needed—both inputs are complete audits with verified line locations.

## Does the released code match the paper?

**Partially. The core math is faithful; the released *default configuration* is not the paper's default.** Anyone who runs `EVON(...)` out of the box gets a whitened Muon-style variant, not the paper's default element-wise-clip EVON.

What matches Algorithm 2 exactly:
- Eigenspace momentum: `exp_avg.mul_(beta1).add_(grad_proj, 1−beta1)` (`evon.py:286`) = line 5.
- Price/Stein Hessian estimator: `f = avg_nxg * (h + wd) * ess` (`evon.py:742-767`) = `Ĥ = E⊙G°/V`, since `1/V = ζ(H+δ)` and `ess`↔ζ, `wd`↔δ. The algebra reconciles cleanly.
- Positivity-correction EMA: `h ← β₂h + (1−β₂)f + ½(1−β₂)²(h−f)²/(h+wd+eps)` = line 6.
- Coupled-WD update: `precond_grad = (exp_avg/debias1 + wd·p_proj)/preconditioner` (`evon.py:313-325`) = line 7.
- Eigenbasis refresh re-projects momentum but not `h_mom` — matches the paper's explicit "H is not updated into the new space."

What does **not** match the paper's default:
- **Whitening is ON by default** (`whiten_prec_grad=True`, `evon.py:58`). `_whiten` (`evon.py:831-836`) runs full Newton-Schulz orthogonalization (`_zeropower_via_newtonschulz`) on every 2-D update every step — a Muon-style spectral *replacement*, not a bounded clip. The paper (Section 3.2) treats spectral clipping as one *optional, mutually-exclusive* choice for line 8's `clip(U)`, an alternative to element-wise clipping.
- **Element-wise clip of the update is absent, and defaulted off where present.** The paper's γ=10 element-wise clip maps to `price_clip_ratio`, which defaults to `None` (off). The one element-wise clip in the code is applied to the *Hessian residual `f`* (`evon.py:762-765`, = paper line 4), not to the update `U` (paper line 8). So with defaults, the code applies *no* clip to the Hessian and *whitening* to the update — neither of the paper's line-8 options.

To reproduce the paper's default EVON: set `whiten_prec_grad=False` and `price_clip_ratio=10`.

## The phasing mechanism & the suspected bug

**Verdict: not a bug.** The suspected clean/noisy phase-parity mismatch between `_sample_params` and `step` is a self-cancelling off-by-one, and it is fully inert by default (`phasing=False`).

The two phase-detection sites use different-looking code:
- Sample site (`_sample_params`, `evon.py:563`): `is_clean_phase = (current_step % 2 == 0) and phasing`, reading `state["step"]` **before** increment.
- Step site (`step`, `evon.py:281-283`): `is_clean_phase = (current_step % 2 != 0 or not sampling_enabled) and phasing`, reading `state["step"]` **after** the `evon.py:273` increment.

The `==0`/`!=0` polarity flip exactly cancels the pre- vs. post-increment offset: `S even ⟺ S+1 odd`. Within one iteration `_sample_params` reads `S`, then `step` increments to `S+1` and reads that, so both resolve to the same phase. The first-encounter `continue` (`evon.py:257-258`) — hit when `Q` is absent and `max_precond_dim>0` — skips the `evon.py:273` increment, creating the permanent `+1` offset that makes the flipped comparisons line up for all later iterations. Decisive trace (phasing=True, sampling on, mc=1): every iteration's noise-applied flag matches the step-site phase, and exactly one of {Hessian update, Shampoo/GG update} fires; iters 1–2 are a benign double-clean warm-up.

Two unrelated asides surfaced during the hunt (neither is the parity bug, both worth flagging to the author only):
- `evon.py:342-350` (`:343`): the `or step % precondition_frequency == 0` clause lets the Shampoo/GG accumulator also refresh on noisy steps — a possible paper-vs-code discrepancy.
- `evon.py:282`'s `or not self._sampling_enabled` term has no analog at the sample site; moot because `sampled_params()` skips `_sample_params` entirely when sampling is off.

## Divergences that matter for us

blrax has **NO phasing** and **NO whitening**, which removes the two most material paper-vs-code gaps from our surface entirely. Ranked by impact:

**A. Affects Bayesian posterior / CL results — the divergences to actually worry about:**
1. **Whitening (`evon.py:58,335-337`).** This is the big one for anyone comparing blrax to the official release, *not* for blrax's own correctness. The official default orthogonalizes every 2-D update every step, replacing the update's spectrum. blrax has no whitening, so blrax implements the paper's *element-wise-clip* default, while stock PyTorch `EVON(...)` implements the *spectral* variant. A naive blrax-vs-official comparison compares two different algorithms. To align, run the official code with `whiten_prec_grad=False`.
2. **Element-wise clip target/default (`evon.py:762-765`, `price_clip_ratio=None`).** The Hessian estimate governs posterior variance `V = 1/(ζ(H+δ))`, so clip semantics on `f` directly shape the posterior. Confirm blrax clips the Hessian residual at the same site and ratio (paper γ=10) as the official code, and that neither silently defaults the clip off.
3. **`ess`/ζ (`evon.py`, required arg).** Directly scales posterior variance. No paper default exists, so this is a per-experiment number that must be matched exactly between blrax and official runs to compare posteriors; it is not a bug, but it is the most likely silent source of posterior mismatch.

**B. Affects only from-scratch optimization speed / trajectory (benign for the posterior):**
4. **Adam-style β₁ bias correction (`evon.py:307-310,324`, `correct_bias=True`).** `debias1 = 1−β₁^step` divides the momentum; material only in the first ~1/(1−β₁) ≈ 20 steps (β₁=0.95), negligible thereafter. `debias2` off by default matches the paper. Check whether blrax debiases identically — a mismatch perturbs only the early trajectory.
5. **Default betas (`evon.py`, `betas=(0.95,0.9999)`, `shampoo_beta=-1`→0.9999).** Not pinned by the paper; unverifiable against it. Match them explicitly across implementations rather than relying on either default.

**C. Inert for us:**
6. **Phasing (`evon.py:275-284,290,336-343,559-573`).** Off by default, not in the paper, and blrax has none. Only relevant if someone enables it in the official code, in which case they leave the paper spec.
7. **Eigenbasis-refresh H re-projection.** Code matches the paper (H not re-projected). Note the repo already patched a related concern in commit `91be154` outside `evon.py`; confirm blrax's refresh matches the paper here.

## Open questions for the author (Adrian)

1. **Whitening default.** Is `whiten_prec_grad=True` the intended default for reproducing the paper's headline numbers, or should the paper's default be element-wise clipping (`whiten_prec_grad=False`, `price_clip_ratio=10`)? Which config generated the paper's main tables?
2. **Which clips are on in the paper's runs?** The paper describes element-wise-vs-spectral clip of the *update* (line 8) as mutually exclusive; the code can apply both the Hessian-residual clip (`price_clip_ratio` on `f`) and update whitening simultaneously. Was that stacking intended, or is `price_clip_ratio` meant only for the non-whitened variant?
3. **Missing line-8 element-wise clip of `U`.** The paper's element-wise clip on the parameter update has no bounded analog in the code (only whitening). Was it dropped deliberately, or does whitening subsume it?
4. **Phasing provenance.** The clean/noisy alternation is not in the paper. Was it used for any published result, or is it purely experimental? Given the duplicated phase predicates (`evon.py:281-283` vs `:563`), would you accept a refactor to a shared helper?
5. **Shampoo/GG refresh on noisy steps (`evon.py:343`).** Is the `or step % precondition_frequency == 0` clause — which refreshes the Shampoo accumulator on noisy steps — intended, or a spec-vs-code discrepancy?
6. **Canonical hyperparameters.** The paper pins only γ=10 and 32 MC samples. Can you provide the per-experiment ζ (`ess`), β₁/β₂/β₃, `hess_init`, T (`precondition_frequency`), α, and WD used for the CL/posterior results so blrax can match them exactly?

Source files: `/private/tmp/claude-501/-Users-conorheins-Documents-evora-ai-research-blrax--claude-worktrees-crazy-torvalds-4da37c/5daca920-f6b2-449e-8b28-c0134a3e4fc2/scratchpad/evon_official/evon-src/evon.py` (paper spec diff and phasing verdict both audit this file).