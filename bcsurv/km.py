"""Kaplan-Meier survival curves and the log-rank test, written out in full."""

import numpy as np
from scipy import stats


def kaplan_meier(time, event):
    """Survival curve S(t) = product over event times t_j <= t of (1 - d_j / n_j).

    d_j is the number of events at t_j and n_j the number still at risk just
    before it. Returns the distinct event times, S at each, and Greenwood's
    variance: Var S = S^2 * sum d_j / (n_j (n_j - d_j)).
    """
    time, event = np.asarray(time, float), np.asarray(event, int)
    times = np.unique(time[event == 1])
    at_risk = np.array([(time >= t).sum() for t in times])
    deaths = np.array([((time == t) & (event == 1)).sum() for t in times])
    surv = np.cumprod(1 - deaths / at_risk)
    with np.errstate(divide="ignore", invalid="ignore"):
        greenwood = np.cumsum(np.where(at_risk > deaths, deaths / (at_risk * (at_risk - deaths)), np.inf))
    variance = np.where(surv > 0, surv**2 * np.where(np.isfinite(greenwood), greenwood, 0.0), 0.0)
    return times, surv, variance


def survival_at(time, event, t):
    """S(t) and its 95% interval on the log(-log) scale, which stays inside (0, 1)."""
    times, surv, var = kaplan_meier(time, event)
    idx = np.searchsorted(times, t, side="right") - 1
    if idx < 0:
        return 1.0, (1.0, 1.0)
    s, v = surv[idx], var[idx]
    if s <= 0 or s >= 1 or not np.isfinite(v):
        return float(s), (float(s), float(s))
    se = np.sqrt(v) / (s * abs(np.log(s)))
    lo = s ** np.exp(1.96 * se)
    hi = s ** np.exp(-1.96 * se)
    return float(s), (float(lo), float(hi))


def logrank(time, event, group):
    """Log-rank test for k groups: chi-square with k - 1 degrees of freedom.

    At each event time, compare each group's observed events with those
    expected if all groups shared one hazard, and combine with the
    hypergeometric covariance.
    """
    time, event, group = np.asarray(time, float), np.asarray(event, int), np.asarray(group)
    levels = np.unique(group)
    k = len(levels)
    observed = np.zeros(k)
    expected = np.zeros(k)
    cov = np.zeros((k, k))
    for t in np.unique(time[event == 1]):
        risk = time >= t
        n = risk.sum()
        d = ((time == t) & (event == 1)).sum()
        n_g = np.array([(risk & (group == g)).sum() for g in levels])
        d_g = np.array([((time == t) & (event == 1) & (group == g)).sum() for g in levels])
        observed += d_g
        expected += d * n_g / n
        if n > 1:
            factor = d * (n - d) / (n**2 * (n - 1))
            cov += factor * (np.diag(n_g * n) - np.outer(n_g, n_g))
    diff = (observed - expected)[:-1]
    chi2 = float(diff @ np.linalg.solve(cov[:-1, :-1], diff))
    return {"chi2": chi2, "df": k - 1, "p": float(stats.chi2.sf(chi2, k - 1)),
            "observed": observed.tolist(), "expected": expected.tolist()}
