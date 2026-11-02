# hawkesflow design

## Model
D event types (here 2: buy-initiated and sell-initiated executions). The intensity of type k is

    λ_k(t) = μ_k(b(t)) + Σ_j Σ_p α[k,j,p] · β_p · Σ_{t_m^j < t} e^{−β_p (t − t_m^j)}

- **Baseline μ_k(b).** Piecewise constant over 13 half-hour bins (09:30–16:00), to absorb the
  intraday U-shape. Without it, seasonality is misread as self-excitation.
- **Kernel.** A sum of P exponentials. α[k,j,p] is the expected number of type-k events
  directly triggered by one type-j event through component p.
- **Branching.** The branching matrix is G = Σ_p α[·,·,p], and the process is stationary iff
  the spectral radius ρ(G) < 1. ρ is the long-run fraction of events that are triggered by
  other events: the endogeneity of the flow.

## Likelihood: O(n·D·P)
    ℓ = Σ_i log λ_{k_i}(t_i) − Σ_k Λ_k(T)

- **Recursion.** R[j,p] = Σ_{past type-j events} e^{−β_p(t − t_m)} decays by e^{−β_p Δt}
  between events and gains 1 at each type-j event.
- **Compensator.** It is closed form: baseline area plus α·(1 − e^{−β(T−t_m)}) per event.
- **Gradient.** It is analytic, in the same pass. The β-derivative uses
  S[j,p] = ∂R[j,p]/∂β_p, which follows S ← e^{−βΔt}(S − Δt·R). The gradient test checks it
  against finite differences to 1e-4 relative.
- **Implementation.** Numba-compiled: 1.8 ms per likelihood-and-gradient evaluation on 53k
  events.

## Fitting
- **Optimiser.** L-BFGS-B on log-parameters, so everything stays positive without
  constraints, using the analytic gradient. Decay rates can be capped (`beta_max`), which
  matters for data with exact ties (see below).
- **Standard errors.** From the inverse observed information, a Hessian by central differences
  of the analytic gradient. The Monte Carlo study (`research/coverage.py`) checks that these
  give nominal 95% coverage.

## Goodness of fit: time rescaling
If the model is right, the compensator increments between consecutive events of each type,
Λ_k(t_i) − Λ_k(t_{i−1}), are iid Exp(1) (Brown et al. 2002). The study reports the KS statistic
(the maximum CDF gap) for each model.
- **Effect size, not significance.** With 20–53k events per symbol every misspecified model is
  formally rejected, so the statistic is used as an effect size.
- **Q-Q plots** show where a model fails: the clustering region versus the long-gap tail.

## Simulation
Ogata thinning, with the bound taken as the current intensity plus the largest baseline ahead,
since kernels only decay between events. It is used for the recovery tests and the coverage
study.

## Data handling: simultaneous executions
- **The mechanism.** One aggressive order that walks several resting orders prints several ITCH
  executions with the **same nanosecond timestamp**.
- **The handling.** `collapse_simultaneous` merges same-timestamp, same-side executions into one
  event. That is the right unit for "a trader decided to cross the spread".
- **Why it matters.** Without collapsing, zero gaps make the likelihood unbounded as β → ∞: no
  MLE exists. With β capped at 1e6/s the fit exists, but it reports excitation that is really
  order fragmentation (see README).

## Not here
Power-law or non-parametric kernels (Bacry-Muzy), marks such as order size, intensity
regressions on the book state, multi-day fits.
