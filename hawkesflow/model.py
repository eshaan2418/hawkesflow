"""Multivariate Hawkes processes with sum-of-exponentials kernels and a piecewise-constant
(intraday-seasonal) baseline.

Intensity of component k:

    lambda_k(t) = mu_k(b(t)) + sum_j sum_p alpha[k, j, p] * beta[p] * sum_{t_m^j < t} exp(-beta[p] (t - t_m^j))

``alpha[k, j, p]`` is the expected number of type-k events directly triggered by one type-j event
through kernel component p (it integrates to alpha), so the branching matrix is
``G = alpha.sum(axis=2)`` and the process is stationary iff its spectral radius is below 1.

The log-likelihood is computed in O(n * D * P) with the standard exponential recursion; the
compensator in closed form. Parameters are fitted by L-BFGS-B on log-transformed values;
standard errors come from the inverse observed information (numerical Hessian).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numba import njit
from scipy import optimize


@dataclass
class HawkesSpec:
    """Model structure: D event types, P kernel components, baseline bins over [0, T)."""

    D: int = 1
    P: int = 1
    bins: np.ndarray = field(default_factory=lambda: np.array([0.0, 1.0]))  # edges, bins[-1] = T

    @property
    def B(self) -> int:
        return len(self.bins) - 1

    def n_params(self) -> int:
        return self.D * self.B + self.D * self.D * self.P + self.P

    def unpack(self, theta: np.ndarray):
        """theta = log of (mu[D, B], alpha[D, D, P], beta[P])."""
        x = np.exp(theta)
        i = self.D * self.B
        mu = x[:i].reshape(self.D, self.B)
        j = i + self.D * self.D * self.P
        alpha = x[i:j].reshape(self.D, self.D, self.P)
        beta = x[j:]
        return mu, alpha, beta

    def pack(self, mu, alpha, beta) -> np.ndarray:
        return np.log(np.concatenate([np.ravel(mu), np.ravel(alpha), np.ravel(beta)]))


@njit(cache=True)
def _loglik(t, k, bins, mu, alpha, beta):
    n = t.shape[0]
    D = mu.shape[0]
    P = beta.shape[0]
    T = bins[-1]
    R = np.zeros((D, P))  # R[j, p] = sum over past type-j events of exp(-beta_p (t - t_m))
    ll = 0.0
    last = 0.0
    b = 0
    for i in range(n):
        dt = t[i] - last
        for p in range(P):
            e = np.exp(-beta[p] * dt)
            for j in range(D):
                R[j, p] *= e
        while b < bins.shape[0] - 2 and t[i] >= bins[b + 1]:
            b += 1
        ki = k[i]
        lam = mu[ki, b]
        for j in range(D):
            for p in range(P):
                lam += alpha[ki, j, p] * beta[p] * R[j, p]
        if lam <= 0.0:
            return -1e300
        ll += np.log(lam)
        R[ki, :] += 1.0
        last = t[i]
    # Compensator: baseline integral + kernel mass of every event (truncated at T).
    comp = 0.0
    for d in range(D):
        for bb in range(bins.shape[0] - 1):
            comp += mu[d, bb] * (bins[bb + 1] - bins[bb])
    for i in range(n):
        j = k[i]
        for p in range(P):
            m = 1.0 - np.exp(-beta[p] * (T - t[i]))
            for d in range(D):
                comp += alpha[d, j, p] * m
    return ll - comp


@njit(cache=True)
def _loglik_grad(t, k, bins, mu, alpha, beta):
    """Log-likelihood and its gradient w.r.t. (mu, alpha, beta) in one O(n D P) pass.

    S[j, p] = dR[j, p]/dbeta_p follows S <- exp(-beta dt) (S - dt R) between events.
    """
    n = t.shape[0]
    D = mu.shape[0]
    P = beta.shape[0]
    B = bins.shape[0] - 1
    T = bins[-1]
    R = np.zeros((D, P))
    S = np.zeros((D, P))
    gmu = np.zeros((D, B))
    ga = np.zeros((D, D, P))
    gb = np.zeros(P)
    ll = 0.0
    last = 0.0
    b = 0
    for i in range(n):
        dt = t[i] - last
        for p in range(P):
            e = np.exp(-beta[p] * dt)
            for j in range(D):
                S[j, p] = e * (S[j, p] - dt * R[j, p])
                R[j, p] *= e
        while b < B - 1 and t[i] >= bins[b + 1]:
            b += 1
        ki = k[i]
        lam = mu[ki, b]
        for j in range(D):
            for p in range(P):
                lam += alpha[ki, j, p] * beta[p] * R[j, p]
        if lam <= 0.0:
            return -1e300, gmu, ga, gb
        inv = 1.0 / lam
        ll += np.log(lam)
        gmu[ki, b] += inv
        for j in range(D):
            for p in range(P):
                ga[ki, j, p] += beta[p] * R[j, p] * inv
                gb[p] += alpha[ki, j, p] * (R[j, p] + beta[p] * S[j, p]) * inv
        R[ki, :] += 1.0
        last = t[i]
    for d in range(D):
        for bb in range(B):
            w = bins[bb + 1] - bins[bb]
            ll -= mu[d, bb] * w
            gmu[d, bb] -= w
    for i in range(n):
        j = k[i]
        for p in range(P):
            tau = T - t[i]
            e = np.exp(-beta[p] * tau)
            for d in range(D):
                ll -= alpha[d, j, p] * (1.0 - e)
                ga[d, j, p] -= 1.0 - e
                gb[p] -= alpha[d, j, p] * tau * e
    return ll, gmu, ga, gb


def loglik_and_grad(spec: HawkesSpec, theta: np.ndarray, t: np.ndarray, k: np.ndarray):
    """Log-likelihood and its gradient with respect to theta (log-parameters)."""
    mu, alpha, beta = spec.unpack(theta)
    ll, gmu, ga, gb = _loglik_grad(t, k, spec.bins, mu, alpha, beta)
    g = np.concatenate([np.ravel(gmu * mu), np.ravel(ga * alpha), gb * beta])  # chain rule: d/dlog x = x d/dx
    return float(ll), g


def loglik(spec: HawkesSpec, theta: np.ndarray, t: np.ndarray, k: np.ndarray) -> float:
    mu, alpha, beta = spec.unpack(theta)
    return float(_loglik(t, k, spec.bins, mu, alpha, beta))


@njit(cache=True)
def _residuals(t, k, bins, mu, alpha, beta):
    """Compensator increments between consecutive events of each type (time rescaling)."""
    n = t.shape[0]
    D = mu.shape[0]
    P = beta.shape[0]
    # Lambda_d(t) evaluated at every event time, accumulated exactly between events.
    Lam = np.zeros(D)
    lastLam = np.full(D, -1.0)
    out = np.empty(n)
    nout = 0
    R = np.zeros((D, P))
    last = 0.0
    b = 0
    for i in range(n):
        # Integrate intensity of every d over (last, t[i]].
        s = last
        while True:
            edge = bins[b + 1] if b < bins.shape[0] - 2 else np.inf
            e_end = min(t[i], edge)
            for d in range(D):
                Lam[d] += mu[d, b] * (e_end - s)
            if t[i] < edge or b >= bins.shape[0] - 2:
                break
            s = edge
            b += 1
        dt = t[i] - last
        for p in range(P):
            decay = 1.0 - np.exp(-beta[p] * dt)
            for j in range(D):
                for d in range(D):
                    Lam[d] += alpha[d, j, p] * R[j, p] * decay
        for p in range(P):
            e = np.exp(-beta[p] * dt)
            for j in range(D):
                R[j, p] *= e
        ki = k[i]
        if lastLam[ki] >= 0.0:
            out[nout] = Lam[ki] - lastLam[ki]
            nout += 1
        lastLam[ki] = Lam[ki]
        R[ki, :] += 1.0
        last = t[i]
    return out[:nout]


def compensator_residuals(spec: HawkesSpec, theta: np.ndarray, t: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Time-rescaled inter-event times; iid Exp(1) if the model is correct."""
    mu, alpha, beta = spec.unpack(theta)
    return _residuals(t, k, spec.bins, mu, alpha, beta)


def _start(spec: HawkesSpec, t, k, beta0):
    T = spec.bins[-1]
    counts = np.array([(k == d).sum() for d in range(spec.D)], dtype=float)
    mu = np.zeros((spec.D, spec.B))
    for d in range(spec.D):
        for b in range(spec.B):
            lo, hi = spec.bins[b], spec.bins[b + 1]
            nb = ((t >= lo) & (t < hi) & (k == d)).sum()
            mu[d, b] = max(nb, 1) / (hi - lo) * 0.5
    alpha = np.full((spec.D, spec.D, spec.P), 0.4 / (spec.D * spec.P))
    beta = np.asarray(beta0, dtype=float)
    del counts, T
    return spec.pack(mu, alpha, beta)


def fit(spec: HawkesSpec, t: np.ndarray, k: np.ndarray, beta0=None, theta0=None, maxiter: int = 500, hessian: bool = True,
        beta_max: float | None = None):
    """Maximum likelihood. Returns dict with theta, loglik, parameters, branching matrix, SEs.

    ``beta_max`` bounds every decay rate (1/s). With exactly simultaneous events (zero gaps) the
    likelihood is unbounded as beta -> infinity and no MLE exists; a bound makes the fit defined.
    """
    t = np.ascontiguousarray(t, dtype=np.float64)
    k = np.ascontiguousarray(k, dtype=np.int64)
    if beta0 is None:
        beta0 = np.logspace(1, -1, spec.P) if spec.P > 1 else np.array([1.0])
    th0 = _start(spec, t, k, beta0) if theta0 is None else theta0
    def fg(th):
        ll, g = loglik_and_grad(spec, th, t, k)
        return -ll, -g
    bounds = [(-50.0, 50.0)] * spec.n_params()
    if beta_max is not None:
        for i in range(spec.n_params() - spec.P, spec.n_params()):
            bounds[i] = (-50.0, float(np.log(beta_max)))
        th0 = np.minimum(th0, [b[1] for b in bounds])
    res = optimize.minimize(fg, th0, jac=True, method="L-BFGS-B", bounds=bounds, options={"maxiter": maxiter, "maxfun": 20000})
    th = res.x
    mu, alpha, beta = spec.unpack(th)
    out = {
        "theta": th,
        "loglik": -res.fun,
        "converged": bool(res.success),
        "iterations": int(res.nit),
        "mu": mu,
        "alpha": alpha,
        "beta": beta,
        "branching": alpha.sum(axis=2),
        "spectral_radius": float(np.max(np.abs(np.linalg.eigvals(alpha.sum(axis=2))))),
        "aic": 2 * spec.n_params() - 2 * (-res.fun),
        "bic": spec.n_params() * np.log(len(t)) - 2 * (-res.fun),
    }
    if hessian:
        H = _num_hessian_from_grad(lambda x: -loglik_and_grad(spec, x, t, k)[1], th)
        try:
            cov = np.linalg.inv(H)
            se_log = np.sqrt(np.clip(np.diag(cov), 0, None))
            out["se_log"] = se_log  # standard errors of log-parameters (delta method: se(x) ~ x se_log)
        except np.linalg.LinAlgError:
            out["se_log"] = None
    return out


def _num_hessian_from_grad(grad, x, h=1e-5):
    """Hessian by central differences of the analytic gradient (2n gradient evaluations)."""
    n = len(x)
    H = np.zeros((n, n))
    for i in range(n):
        xp, xm = x.copy(), x.copy()
        xp[i] += h
        xm[i] -= h
        H[:, i] = (grad(xp) - grad(xm)) / (2 * h)
    return 0.5 * (H + H.T)


def _num_hessian(f, x, h=1e-4):
    n = len(x)
    H = np.zeros((n, n))
    f0 = f(x)
    for i in range(n):
        for j in range(i, n):
            xi = x.copy()
            if i == j:
                xp, xm = x.copy(), x.copy()
                xp[i] += h
                xm[i] -= h
                H[i, i] = (f(xp) - 2 * f0 + f(xm)) / h**2
            else:
                xpp, xpm, xmp, xmm = x.copy(), x.copy(), x.copy(), x.copy()
                xpp[i] += h; xpp[j] += h
                xpm[i] += h; xpm[j] -= h
                xmp[i] -= h; xmp[j] += h
                xmm[i] -= h; xmm[j] -= h
                H[i, j] = H[j, i] = (f(xpp) - f(xpm) - f(xmp) + f(xmm)) / (4 * h * h)
            del xi
    return H


@njit(cache=True)
def _simulate(bins, mu, alpha, beta, seed):
    np.random.seed(seed)
    D = mu.shape[0]
    P = beta.shape[0]
    T = bins[-1]
    cap = 10_000_000
    ts = np.empty(cap)
    ks = np.empty(cap, dtype=np.int64)
    n = 0
    R = np.zeros((D, P))
    t = 0.0
    b = 0
    while True:
        # Upper bound: current intensity is maximal now (kernels only decay until the next event),
        # but the baseline may step up at a bin edge, so bound by the max baseline ahead.
        lam_bar = 0.0
        for d in range(D):
            m = 0.0
            for bb in range(b, bins.shape[0] - 1):
                m = max(m, mu[d, bb])
            lam_bar += m
            for j in range(D):
                for p in range(P):
                    lam_bar += alpha[d, j, p] * beta[p] * R[j, p]
        w = np.random.exponential(1.0 / lam_bar)
        t_new = t + w
        if t_new >= T:
            break
        for p in range(P):
            e = np.exp(-beta[p] * w)
            for j in range(D):
                R[j, p] *= e
        t = t_new
        while b < bins.shape[0] - 2 and t >= bins[b + 1]:
            b += 1
        lam = np.empty(D)
        tot = 0.0
        for d in range(D):
            x = mu[d, b]
            for j in range(D):
                for p in range(P):
                    x += alpha[d, j, p] * beta[p] * R[j, p]
            lam[d] = x
            tot += x
        u = np.random.random() * lam_bar
        if u < tot:
            acc = 0.0
            kk = D - 1
            for d in range(D):
                acc += lam[d]
                if u < acc:
                    kk = d
                    break
            if n >= cap:
                break
            ts[n] = t
            ks[n] = kk
            n += 1
            R[kk, :] += 1.0
    return ts[:n], ks[:n]


def simulate(spec: HawkesSpec, theta: np.ndarray, seed: int = 0):
    """Ogata thinning. Returns (times, types)."""
    mu, alpha, beta = spec.unpack(theta)
    return _simulate(spec.bins, mu, alpha, beta, seed)


def collapse_simultaneous(t: np.ndarray, k: np.ndarray, size: np.ndarray | None = None):
    """Merge events with identical timestamp and type into one (one aggressive order that executes
    against several resting orders prints several executions at the same nanosecond)."""
    if len(t) == 0:
        return t, k, size
    keep = np.ones(len(t), dtype=bool)
    keep[1:] = ~((t[1:] == t[:-1]) & (k[1:] == k[:-1]))
    idx = np.flatnonzero(keep)
    if size is None:
        return t[idx], k[idx], None
    agg = np.add.reduceat(size, idx)
    return t[idx], k[idx], agg
