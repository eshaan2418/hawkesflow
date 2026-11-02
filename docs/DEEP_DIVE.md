# hawkesflow: deep-dive interview questions

### 1. What is a Hawkes process, in one sentence?
A point process whose intensity jumps after each event and decays back. Events beget events, and
the expected number of direct offspring per event, the branching ratio, measures how much of the
activity is self-generated.

### 2. Why is the log-likelihood O(n) for exponential kernels?
The sum over past events, Σ e^{−β(t−t_m)}, satisfies a one-step recursion: multiply by
e^{−βΔt}, then add 1 at each event. With D types and P components the cost is O(n·D·P), with no
quadratic sum over pairs. The compensator is closed form.

### 3. How do you get the gradient with respect to β cheaply?
- **Recursion.** Differentiate the recursion: S = ∂R/∂β obeys S ← e^{−βΔt}(S − Δt·R) between
  events.
- **Gradient terms.** The contribution to ∂ℓ/∂β is α(R + β·S)/λ at each event, minus
  α(T−t)e^{−β(T−t)} from the compensator.
- **Payoff.** Fits became 37× faster (5.7 s → 0.15 s) and reached equal or higher likelihoods.

### 4. How do you know the fit is good?
- **Time rescaling.** If the model is right, the compensator increments between events are iid
  Exp(1). This works for any point process (Brown et al. 2002), and the KS distance and Q-Q
  plots measure the deviation.
- **Results.** Seasonal Poisson: KS 0.33–0.57. Hawkes with 3 exponentials: 0.014–0.051.
- **Caveat.** With tens of thousands of events every model is formally rejected, so the size of
  the deviation is what matters.

### 5. How do you know the estimator itself is right?
- **Unit tests.** Simulate from known parameters with Ogata thinning and refit. Recovery is
  within 4 standard errors (tests).
- **Monte Carlo.** 40 days simulated at AAPL's fitted parameters (about 52,900 events each). The
  95% Wald intervals covered the truth 94.75% (kernel) and 95.5% (baseline) of the time, and the
  spectral radius was recovered as 0.5277 ± 0.0025 against a true 0.5283.
- **Brute force.** The likelihood matches an O(n²) brute force to 1e-8.

### 6. Why a seasonal baseline?
Activity is U-shaped over the day (open, lunch lull, close). A constant baseline forces the
kernel to explain slow, predictable variation as self-excitation and inflates the branching
ratio. Thirteen half-hour bins absorb it. This is the "spurious endogeneity from
non-stationarity" problem.

### 7. What went wrong with raw ITCH executions?
- **Shared timestamps.** One aggressive order that hits five resting orders prints five
  executions with the same timestamp. Zero inter-event gaps let λ at those points go to
  infinity as β → ∞, so the likelihood is unbounded and no MLE exists.
- **Capped fit.** With β capped at 1e6/s, the spectral radius rises from 0.41–0.65 (collapsed)
  to 0.65–0.72: fragmentation reads as endogeneity.
- **The fix** is to define the event as one aggressive order (same timestamp and side).

### 8. What do the timescales mean?
The 2-exponential fits put excitation at about 0.2 ms and 6–23 ms. The 3-exponential fits add
50–250 µs, about 1–8 ms and 27 ms–2 s. Adding components keeps improving BIC, which is what a
power-law kernel looks like when approximated by exponentials. Microsecond reactions are
co-located algorithms responding to the feed. Milliseconds to seconds are order slicing and
human-scale reaction.

### 9. Self-excitation vs cross-excitation: what does it tell you?
Buys trigger buys (0.33–0.61), and sells trigger sells. Buy↔sell cross-excitation is small
(0.04–0.09). This points to order splitting and momentum-style reactions dominating, rather
than liquidity provision immediately reversing flow. That sign asymmetry is what makes order
flow persistent (Lillo-Farmer) while prices stay roughly efficient.

### 10. Why is the spectral radius the right "endogeneity" measure?
In a stationary multivariate Hawkes process the mean intensity is (I − G)^{-1}·μ̄. The fraction
of events with an ancestor among other events is governed by G's largest eigenvalue: a cluster
contains on average 1/(1 − ρ) events. ρ → 1 is the critical regime, where tiny exogenous input
produces huge activity. Some studies report liquid markets drifting toward that limit.

### 11. Why L-BFGS-B on log-parameters?
Positivity comes for free and the problem is better conditioned: rates span 1e-1 to 1e4. Bounds
are only needed to cap β. The Hessian for standard errors is taken in log space too, and
delta-method standard errors follow.

### 12. How would you extend this?
- Power-law or non-parametric kernels (Bacry-Muzy Wiener-Hopf).
- Marks (order size, aggressiveness) and state dependence (spread, imbalance).
- A doubly stochastic baseline for the long-gap misfit.
- Multi-asset excitation (QQQ → constituents) for lead-lag.
- Using the fitted intensity as a short-horizon activity forecast in execution scheduling.
