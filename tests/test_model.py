import numpy as np
from scipy import stats

from hawkesflow import HawkesSpec, collapse_simultaneous, compensator_residuals, fit, loglik, simulate


def brute_loglik(t, k, bins, mu, alpha, beta):
    D, P = mu.shape[0], beta.shape[0]
    T = bins[-1]
    ll = 0.0
    for i in range(len(t)):
        b = np.searchsorted(bins, t[i], side="right") - 1
        lam = mu[k[i], b]
        for m in range(i):
            for p in range(P):
                lam += alpha[k[i], k[m], p] * beta[p] * np.exp(-beta[p] * (t[i] - t[m]))
        ll += np.log(lam)
    comp = sum(mu[d, b] * (bins[b + 1] - bins[b]) for d in range(D) for b in range(len(bins) - 1))
    for m in range(len(t)):
        for d in range(D):
            for p in range(P):
                comp += alpha[d, k[m], p] * (1 - np.exp(-beta[p] * (T - t[m])))
    return ll - comp


def test_loglik_matches_brute_force():
    rng = np.random.default_rng(0)
    spec = HawkesSpec(D=2, P=2, bins=np.array([0.0, 3.0, 7.0, 10.0]))
    t = np.sort(rng.uniform(0, 10, 150))
    k = rng.integers(0, 2, 150)
    mu = rng.uniform(0.5, 2, (2, 3))
    alpha = rng.uniform(0.05, 0.3, (2, 2, 2))
    beta = np.array([8.0, 0.7])
    th = spec.pack(mu, alpha, beta)
    assert abs(loglik(spec, th, t, k) - brute_loglik(t, k, spec.bins, mu, alpha, beta)) < 1e-8


def test_univariate_recovery_and_residuals():
    spec = HawkesSpec(D=1, P=1, bins=np.array([0.0, 4000.0]))
    true = spec.pack(np.array([[1.0]]), np.array([[[0.6]]]), np.array([5.0]))
    t, k = simulate(spec, true, seed=3)
    r = fit(spec, t, k, beta0=[2.0])
    se = r["se_log"]
    for est, tru, s in zip(r["theta"], true, se):
        assert abs(est - tru) < 4 * s, (np.exp(r["theta"]), np.exp(true), se)
    res = compensator_residuals(spec, r["theta"], t, k)
    assert stats.kstest(res, "expon").pvalue > 0.01
    # A Poisson model (alpha ~ 0) is rejected on the same data.
    pois = spec.pack(np.array([[len(t) / 4000.0]]), np.array([[[1e-9]]]), np.array([5.0]))
    assert stats.kstest(compensator_residuals(spec, pois, t, k), "expon").pvalue < 1e-6


def test_bivariate_cross_excitation_recovered():
    spec = HawkesSpec(D=2, P=1, bins=np.array([0.0, 3000.0]))
    alpha = np.array([[[0.3]], [[0.25]]])  # row = receiver: buy<-buy 0.3, sell<-buy 0.25
    alpha = np.array([[[0.30], [0.10]], [[0.25], [0.35]]])
    true = spec.pack(np.array([[0.8], [0.6]]), alpha, np.array([3.0]))
    t, k = simulate(spec, true, seed=11)
    r = fit(spec, t, k, beta0=[1.0])
    G = r["branching"]
    assert np.all(np.abs(G - alpha.sum(axis=2)) < 0.06), G
    assert r["spectral_radius"] < 1


def test_seasonal_baseline():
    bins = np.array([0.0, 1000.0, 2000.0, 3000.0])
    spec = HawkesSpec(D=1, P=1, bins=bins)
    true = spec.pack(np.array([[3.0, 0.5, 2.0]]), np.array([[[0.4]]]), np.array([4.0]))
    t, k = simulate(spec, true, seed=5)
    r = fit(spec, t, k, beta0=[2.0])
    assert np.allclose(r["mu"][0], [3.0, 0.5, 2.0], rtol=0.15), r["mu"]
    res = compensator_residuals(spec, r["theta"], t, k)
    assert stats.kstest(res, "expon").pvalue > 0.01


def test_collapse():
    t = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 3.0])
    k = np.array([0, 0, 1, 1, 1, 0])
    s = np.array([100, 200, 50, 10, 20, 5])
    tc, kc, sc = collapse_simultaneous(t, k, s)
    assert tc.tolist() == [1.0, 1.0, 2.0, 3.0] and kc.tolist() == [0, 1, 1, 0] and sc.tolist() == [300, 50, 30, 5]


def test_analytic_gradient_matches_finite_differences():
    from hawkesflow import loglik_and_grad

    rng = np.random.default_rng(1)
    spec = HawkesSpec(D=2, P=2, bins=np.array([0.0, 40.0, 100.0]))
    t = np.sort(rng.uniform(0, 100, 600))
    k = rng.integers(0, 2, 600)
    th = spec.pack(rng.uniform(0.5, 2, (2, 2)), rng.uniform(0.05, 0.3, (2, 2, 2)), np.array([6.0, 0.4]))
    ll, g = loglik_and_grad(spec, th, t, k)
    assert abs(ll - loglik(spec, th, t, k)) < 1e-9
    h = 1e-6
    for i in range(len(th)):
        e = np.zeros_like(th)
        e[i] = h
        fd = (loglik(spec, th + e, t, k) - loglik(spec, th - e, t, k)) / (2 * h)
        assert abs(fd - g[i]) < 1e-4 * max(1, abs(fd)), (i, fd, g[i])
