"""Monte Carlo check of the estimator: simulate from the model fitted to AAPL, refit, and measure
bias and 95% confidence-interval coverage (Wald intervals on log-parameters from the observed
information).

usage: python research/coverage.py ITCH_FITS_JSON OUT_JSON [REPLICATIONS]
"""

from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from hawkesflow import HawkesSpec, fit, simulate

T = 23400.0
BINS = np.linspace(0.0, T, 14)


def one(args):
    theta, seed = args
    spec = HawkesSpec(D=2, P=2, bins=BINS)
    t, k = simulate(spec, np.array(theta), seed=seed)
    r = fit(spec, t, k, beta0=[100.0, 1.0], hessian=True)
    return {"theta": r["theta"].tolist(), "se_log": None if r["se_log"] is None else r["se_log"].tolist(),
            "n": len(t), "rho": r["spectral_radius"], "converged": r["converged"]}


def main():
    with open(sys.argv[1]) as fh:
        fits = json.load(fh)
    reps = int(sys.argv[3]) if len(sys.argv) > 3 else 30
    aapl = next(s for s in fits["symbols"] if s["symbol"] == "AAPL")["models"]["hawkes_2exp"]
    # Reconstruct the AAPL fit's parameters from the saved values.
    spec = HawkesSpec(D=2, P=2, bins=BINS)
    alpha = np.array(aapl["alpha_by_component"])
    beta = 1.0 / np.array(aapl["timescales_s"])
    mu = np.array(aapl["mu"])
    true = spec.pack(mu, alpha, beta)
    with ProcessPoolExecutor(max_workers=8) as ex:
        res = list(ex.map(one, [(true.tolist(), 1000 + i) for i in range(reps)]))
    names = [f"mu[{d},{b}]" for d in range(2) for b in range(13)] + \
        [f"alpha[{i},{j},{p}]" for i in range(2) for j in range(2) for p in range(2)] + ["beta[0]", "beta[1]"]
    est = np.array([r["theta"] for r in res])
    se = np.array([r["se_log"] for r in res if r["se_log"] is not None])
    cover = np.abs(est[: len(se)] - true) <= 1.96 * se
    out = {"replications": reps, "events_mean": float(np.mean([r["n"] for r in res])),
           "true_spectral_radius": aapl["spectral_radius"], "est_spectral_radius_mean": float(np.mean([r["rho"] for r in res])),
           "est_spectral_radius_sd": float(np.std([r["rho"] for r in res], ddof=1)),
           "coverage_95_by_param": {n: float(c) for n, c in zip(names, cover.mean(axis=0))},
           "coverage_95_mean_over_kernel_params": float(cover[:, 26:].mean()),
           "coverage_95_mean_over_baseline_params": float(cover[:, :26].mean()),
           "relative_bias_kernel": {n: float(np.mean(np.exp(est[:, i] - true[i]) - 1)) for i, n in enumerate(names) if i >= 26}}
    with open(sys.argv[2], "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "coverage_95_by_param"}, indent=1))


if __name__ == "__main__":
    main()
