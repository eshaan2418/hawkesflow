"""Timing: log-likelihood evaluation and full fits on AAPL-sized synthetic data.

usage: python research/bench.py OUT_JSON
"""

import json
import sys
import time

import numpy as np

from hawkesflow import HawkesSpec, fit, loglik, simulate

T = 23400.0
spec = HawkesSpec(D=2, P=2, bins=np.linspace(0.0, T, 14))
mu = np.full((2, 13), 0.55)
alpha = np.array([[[0.25, 0.2], [0.03, 0.03]], [[0.03, 0.04], [0.24, 0.2]]])
theta = spec.pack(mu, alpha, np.array([4000.0, 130.0]))
t0 = time.perf_counter()
t, k = simulate(spec, theta, seed=1)
sim_s = time.perf_counter() - t0
loglik(spec, theta, t, k)  # JIT warm-up
n = 50
t0 = time.perf_counter()
for _ in range(n):
    loglik(spec, theta, t, k)
ll_s = (time.perf_counter() - t0) / n
t0 = time.perf_counter()
r = fit(spec, t, k, beta0=[100.0, 1.0], hessian=False)
fit_s = time.perf_counter() - t0
t0 = time.perf_counter()
fit(spec, t, k, beta0=[100.0, 1.0], hessian=True)
fit_h_s = time.perf_counter() - t0
out = {"events": len(t), "simulate_s": sim_s, "loglik_ms": ll_s * 1e3, "loglik_events_per_s": len(t) / ll_s,
       "fit_s": fit_s, "fit_iterations": r["iterations"], "fit_with_hessian_s": fit_h_s, "params": spec.n_params()}
with open(sys.argv[1], "w") as fh:
    json.dump(out, fh, indent=1)
print(out)
