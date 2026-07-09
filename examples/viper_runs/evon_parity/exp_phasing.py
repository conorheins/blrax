"""E4. Instrument the official EVON 'phasing' mode to test Dimitrije's suspected bug.

Phasing is supposed to ALTERNATE:
  clean step: no noise, update L/R preconditioner (GG) from the mode gradient, freeze H
  noisy step: add posterior noise, update the diagonal Hessian H (Price), freeze GG

The parity is decided in two places reading the step counter at different times
(_sample_params BEFORE the increment, step() AFTER). We log, per optimizer iteration:
  - phase assigned at SAMPLING time (_sample_params)  -> governs whether noise is added
  - phase assigned at UPDATE time   (step())          -> governs H vs GG update
  - whether H (h_mom) actually changed this step
  - whether the preconditioner accumulator (GG) actually changed this step
and check the intended invariant: exactly one of {H, GG} updates each step, and the
noise-on parity matches the H-update parity.
"""
import sys
import numpy as np
import torch

sys.path.insert(0, __file__.rsplit('/', 1)[0] + '/evon_official/evon-src')
import evon as evon_mod
from evon import EVON

torch.manual_seed(0); np.random.seed(0)

# --- spy on the sampling-time phase decision (H/GG changes detected via snapshots) ---
LOG = []
_orig_sample = EVON._sample_params

def sample_spy(self, train=True):
    ev = {}
    for group in self.param_groups:
        for p in group["params"]:
            cs = self.state[p].get("step", 0)
            ev['sample_step'] = cs
            ev['sample_clean'] = bool((cs % 2 == 0) and group.get("phasing", self._phasing))
    result = _orig_sample(self, train=train)
    # read noise NOW, before _restore_param_means pops self._noises
    nz = sum(float(self._noises[id(p)].norm())
             for p in self.param_groups[0]["params"] if id(p) in self._noises)
    ev['noise_norm'] = nz
    LOG.append(('sample', ev))
    return result

EVON._sample_params = sample_spy

# --- tiny run with phasing=True ---
d_in, d_out, B = 6, 4, 16
X = torch.randn(B, d_in); Y = torch.randint(0, d_out, (B,))
lin = torch.nn.Linear(d_in, d_out)
opt = EVON(lin.parameters(), ess=50.0, hess_init=1.0, lr=0.05,
           betas=(0.9, 0.99), shampoo_beta=0.95, weight_decay=1e-6,
           precondition_frequency=100,     # keep Q refresh out of the way
           max_precond_dim=10000, precondition_1d=False, correct_bias=True,
           phasing=True, whiten_prec_grad=False, mc_samples=1)
crit = torch.nn.CrossEntropyLoss()

# snapshot helpers on the weight param's state
wp = lin.weight

def snap():
    st = opt.state[wp]
    h = st.get("h_mom"); gg = st.get("GG")
    hs = None if h is None else float(h.sum())
    ggs = None if not gg or (len(gg) and (len(gg[0]) == 0)) else float(sum(
        (g.sum() if hasattr(g, 'sum') and len(g) > 0 else 0.0) for g in gg))
    return hs, ggs

print("iter | sample_step sample_phase | update_step update_phase | noise? | H changed | GG changed")
print("-" * 96)
rows = []
for it in range(1, 9):
    LOG.clear()
    h0, gg0 = snap()
    with opt.sampled_params(train=True):
        opt.zero_grad()
        loss = crit(lin(X), Y)
        loss.backward()
    # capture the sampling-time decision + noise norm (recorded inside the hook)
    samp = [e for k, e in LOG if k == 'sample'][-1]
    noise_norm = samp.get('noise_norm', 0.0)
    opt.step()
    h1, gg1 = snap()
    us = opt.state[wp]["step"]
    upd_clean = bool((us % 2 != 0) and True)     # step()'s clean rule (phasing on)
    h_changed = (h0 is not None and h1 is not None and abs(h1 - h0) > 1e-9)
    gg_changed = (gg0 is not None and gg1 is not None and abs((gg1 or 0) - (gg0 or 0)) > 1e-12)
    rows.append((it, samp['sample_step'], samp['sample_clean'], us, upd_clean,
                 noise_norm > 1e-9, h_changed, gg_changed))
    print(f"{it:4d} | {samp['sample_step']:11d} {'CLEAN' if samp['sample_clean'] else 'noisy':>12s} |"
          f" {us:11d} {'CLEAN' if upd_clean else 'noisy':>12s} |"
          f" {'yes' if noise_norm>1e-9 else 'no ':>6s} |"
          f" {'yes' if h_changed else 'no ':>9s} | {'yes' if gg_changed else 'no ':>10s}")

print()
# invariant checks
viol = []
for (it, ss, sc, us, uc, noise_on, hch, ggch) in rows:
    if it == 1:      # first iter is the Shampoo-init skip; special-case
        continue
    # invariant A: noise applied  iff  update phase is NOISY (H should update)
    if noise_on != (not uc):
        viol.append(f"iter{it}: noise_on={noise_on} but update_phase={'clean' if uc else 'noisy'} (parity mismatch)")
    # invariant B: exactly one of H/GG updates on a phasing step
    if hch and ggch:
        viol.append(f"iter{it}: BOTH H and GG updated (should be exactly one)")
    if (not hch) and (not ggch):
        viol.append(f"iter{it}: NEITHER H nor GG updated")
    # invariant C: H updates <=> noisy update phase
    if hch and uc:
        viol.append(f"iter{it}: H updated on a CLEAN phase")
    if noise_on and not hch:
        viol.append(f"iter{it}: noisy sample but H did NOT update (lost Hessian info)")

print("INVARIANT VIOLATIONS:" if viol else "No invariant violations detected in the default phasing trace.")
for v in viol:
    print("  -", v)
print()
print("Note: blrax has NO phasing mode — its EVON updates BOTH the Hessian (Price/")
print("sampling) and the L/R covariance every step from the same noisy sample, so this")
print("clean/noisy alternation (and any bug in it) does not exist on the blrax side.")
