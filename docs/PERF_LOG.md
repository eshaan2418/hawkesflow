# hawkesflow performance log

Machine: Apple M5, Python 3.12, numba 0.68, shared with another workload (`bench/results/ENV.txt`).

| | finite-difference gradient (scipy) | analytic gradient | source |
|---|---|---|---|
| fit, 52,961 events, 36 parameters | 5.69 s | **0.153 s** (37×) | `bench_numeric_gradient.json` / `bench.json` |
| fit + Hessian (standard errors) | 11.6 s | **0.32 s** | same |
| one likelihood (+ gradient) | 2.08 ms | 1.78 ms | same |
| simulate one AAPL-sized day (Ogata) | 0.09 s | 0.08 s | same |

- **Why it was slow.** L-BFGS-B with a finite-difference gradient evaluates the likelihood 37
  times per gradient.
- **The fix.** The analytic gradient costs about the same as one likelihood: one pass with an
  extra recursion for ∂R/∂β. The Hessian (central differences of the gradient, 2n gradient
  calls) replaced 4·n² likelihood calls.
- **Effect on results.** Every fitted log-likelihood was equal or higher (up to +4.5 nats on the
  hard uncollapsed fits), so the old optimiser had sometimes stopped slightly short.
- **The whole study** (6 symbols × 6 models) now runs in under 3 s of fitting per symbol.

## Correctness journey
| issue | how found | fix |
|---|---|---|
| NaN log-likelihood on raw executions | fit output | real: no MLE with zero gaps (β → ∞). Cap β (1e6/s) and report it as a diagnostic, not a model |
| Patch anchored on a lint-removed comment left `fg` undefined | test failures | patch the right line |
| Package forgot to export `loglik_and_grad` | test import error | export it |
| Baseline μ not saved, needed for the coverage study | study code | save μ in the results |
