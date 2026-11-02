"""Fit Poisson and Hawkes models to NASDAQ aggressor order flow (ITCH 2019-01-30, 09:30-16:00).

usage: python research/fit_itch.py DATA_DIR OUT_JSON [SYMBOL ...]
"""

from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy import stats

from hawkesflow import HawkesSpec, collapse_simultaneous, compensator_residuals, fit

T = 23400.0  # 09:30-16:00 in seconds
BINS = np.linspace(0.0, T, 14)  # 13 half-hour baseline bins


def load(path: Path):
    a = np.loadtxt(path, delimiter=",", dtype=np.float64)
    order = np.argsort(a[:, 0], kind="stable")
    a = a[order]
    t = (a[:, 0] - 34200e9) / 1e9
    k = np.where(a[:, 1] > 0, 0, 1).astype(np.int64)  # 0 = buy aggressor, 1 = sell aggressor
    return t, k, a[:, 2]


def poisson(t, k, bins):
    """MLE of a piecewise-constant Poisson baseline (closed form) and its residual KS."""
    D, B = 2, len(bins) - 1
    mu = np.zeros((D, B))
    for d in range(D):
        for b in range(B):
            mu[d, b] = max(((t >= bins[b]) & (t < bins[b + 1]) & (k == d)).sum(), 1) / (bins[b + 1] - bins[b])
    spec = HawkesSpec(D=D, P=1, bins=bins)
    th = spec.pack(mu, np.full((2, 2, 1), 1e-12), np.array([1.0]))
    from hawkesflow import loglik

    ll = loglik(spec, th, t, k)
    n_par = D * B
    return spec, th, ll, n_par


def summarize(spec, r, t, k):
    res = compensator_residuals(spec, r["theta"], t, k)
    ksr = stats.kstest(res, "expon")
    lam_bar = len(t) / T
    mu_bar = r["mu"].mean(axis=1).sum()
    return {
        "loglik": r["loglik"], "aic": r["aic"], "bic": r["bic"], "converged": r["converged"],
        "mu": r["mu"].tolist(), "branching": np.round(r["branching"], 4).tolist(), "spectral_radius": r["spectral_radius"],
        "timescales_s": (1.0 / r["beta"]).tolist(),
        "alpha_by_component": np.round(r["alpha"], 4).tolist(),
        "endogenous_share": 1.0 - mu_bar / lam_bar,
        "ks_statistic": float(ksr.statistic), "ks_pvalue": float(ksr.pvalue),
        "residual_mean": float(res.mean()), "residual_var": float(res.var()),
        "n_params": spec.n_params(),
    }


def run_symbol(args):
    path = Path(args)
    sym = path.stem
    t0 = time.time()
    t_raw, k_raw, size = load(path)
    t, k, _ = collapse_simultaneous(t_raw, k_raw, size)
    # Ties between a buy and a sell at the same nanosecond remain; the likelihood allows them.
    out = {"symbol": sym, "events_raw": len(t_raw), "events_collapsed": len(t),
           "buy_share": float((k == 0).mean())}
    models = {}
    for name, bins in (("poisson_const", np.array([0.0, T])), ("poisson_seasonal", BINS)):
        spec, th, ll, npar = poisson(t, k, bins)
        res = compensator_residuals(spec, th, t, k)
        kst = stats.kstest(res, "expon")
        models[name] = {"loglik": ll, "aic": 2 * npar - 2 * ll, "bic": npar * np.log(len(t)) - 2 * ll,
                        "ks_statistic": float(kst.statistic), "ks_pvalue": float(kst.pvalue), "n_params": npar}
    for P, beta0 in ((1, [10.0]), (2, [100.0, 1.0]), (3, [500.0, 20.0, 0.5])):
        spec = HawkesSpec(D=2, P=P, bins=BINS)
        r = fit(spec, t, k, beta0=beta0, hessian=(P == 2))
        models[f"hawkes_{P}exp"] = summarize(spec, r, t, k)
        if P == 2 and r.get("se_log") is not None:
            se = r["se_log"]
            i0 = spec.D * spec.B
            a = r["alpha"].ravel()
            models["hawkes_2exp"]["alpha_se"] = np.round(a * se[i0:i0 + a.size], 4).reshape(r["alpha"].shape).tolist()
    # The same 2-exp model on uncollapsed executions: how much excitation is an artefact? Zero
    # gaps make the likelihood unbounded as beta -> inf, so decay rates are capped at 1e6/s (1 us).
    spec = HawkesSpec(D=2, P=2, bins=BINS)
    rr = fit(spec, t_raw, k_raw, beta0=[100.0, 1.0], hessian=False, beta_max=1e6)
    models["hawkes_2exp_uncollapsed"] = summarize(spec, rr, t_raw, k_raw)
    out["models"] = models
    out["seconds"] = time.time() - t0
    return out


def main():
    data, out = Path(sys.argv[1]), sys.argv[2]
    syms = sys.argv[3:] or ["AAPL", "MSFT", "AMZN", "NVDA", "TSLA", "QQQ"]
    paths = [str(data / f"{s}.csv") for s in syms]
    with ProcessPoolExecutor(max_workers=min(6, len(paths))) as ex:
        results = list(ex.map(run_symbol, paths))
    Path(out).write_text(json.dumps({"date": "2019-01-30", "session": "09:30-16:00", "baseline_bins": 13,
                                     "symbols": results}, indent=1))
    for r in results:
        m = r["models"]
        print(r["symbol"], r["events_raw"], r["events_collapsed"],
              "KS pois", round(m["poisson_seasonal"]["ks_statistic"], 3),
              "KS h2", round(m["hawkes_2exp"]["ks_statistic"], 3),
              "rho h2", round(m["hawkes_2exp"]["spectral_radius"], 3),
              "rho raw", round(m["hawkes_2exp_uncollapsed"]["spectral_radius"], 3),
              "ts", [round(x, 4) for x in m["hawkes_2exp"]["timescales_s"]], round(r["seconds"]), "s")


if __name__ == "__main__":
    main()
