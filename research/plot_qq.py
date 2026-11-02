"""Q-Q plot of time-rescaled residuals (AAPL): Exp(1) quantiles vs empirical, per model.

usage: python research/plot_qq.py DATA_DIR OUT_PNG
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from fit_itch import BINS, load, poisson

from hawkesflow import HawkesSpec, collapse_simultaneous, compensator_residuals, fit


def main():
    t, k, s = load(Path(sys.argv[1]) / "AAPL.csv")
    t, k, _ = collapse_simultaneous(t, k, s)
    q = np.linspace(0.001, 0.999, 400)
    theo = -np.log1p(-q)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    spec, th, _, _ = poisson(t, k, BINS)
    curves = [("Poisson, seasonal baseline", spec, th, "#9e9e9e")]
    for P, b0, c in ((1, [10.0], "#4c78a8"), (2, [100.0, 1.0], "#f58518"), (3, [500.0, 20.0, 0.5], "#54a24b")):
        sp = HawkesSpec(D=2, P=P, bins=BINS)
        r = fit(sp, t, k, beta0=b0, hessian=False)
        curves.append((f"Hawkes, {P} exponential{'s' if P > 1 else ''}", sp, r["theta"], c))
    for name, sp, th, c in curves:
        res = compensator_residuals(sp, th, t, k)
        emp = np.quantile(res, q)
        for a in ax:
            a.plot(theo, emp, label=name, color=c, lw=1.8)
    for a, lim in zip(ax, (8, 1.0)):
        a.plot([0, lim], [0, lim], color="black", lw=0.8, ls="--", label="Exp(1) (perfect fit)")
        a.set_xlim(0, lim)
        a.set_ylim(0, lim * (1.6 if lim > 1 else 2.5))
        a.set_xlabel("Exp(1) quantile")
        a.grid(alpha=0.3)
    ax[0].set_ylabel("residual quantile")
    ax[0].set_title("AAPL aggressor order flow, 2019-01-30: time-rescaled residuals")
    ax[1].set_title("zoom: short residuals (clustering)")
    ax[0].legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(sys.argv[2], dpi=130)
    print("wrote", sys.argv[2])


if __name__ == "__main__":
    main()
