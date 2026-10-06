"""The Cox proportional hazards model, fitted from first principles.

The model says each woman's hazard is a shared baseline hazard times
exp(x . beta). Cox's partial likelihood removes the baseline: at each event
time, the chance that it was this woman, out of everyone still at risk, is

    exp(x_i . beta) / sum over the risk set of exp(x_j . beta).

beta maximises the product of these over all events (Breslow's handling of
tied times). The log of it is concave, so Newton-Raphson converges quickly:
beta <- beta + I(beta)^-1 U(beta), with U the gradient (score) and I the
negative Hessian (information). The inverse information also gives the
standard errors.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy import stats


@dataclass
class CoxModel:
    names: list
    beta: np.ndarray
    se: np.ndarray
    means: np.ndarray
    loglik: float
    loglik_null: float
    iterations: int
    base_times: np.ndarray = field(repr=False)
    base_cumhaz: np.ndarray = field(repr=False)

    def linear_predictor(self, X):
        """x . beta, centred on the development cohort's means."""
        return (np.asarray(X, float) - self.means) @ self.beta

    def survival(self, X, t):
        """S(t | x) = S0(t) ^ exp(lp), with S0 from Breslow's estimator."""
        h0 = np.interp(t, self.base_times, self.base_cumhaz, left=0.0)
        return np.exp(-h0 * np.exp(self.linear_predictor(X)))

    def risk(self, X, t):
        return 1 - self.survival(X, t)

    def table(self):
        z = self.beta / self.se
        return [{
            "name": n, "beta": float(b), "se": float(s), "hr": float(np.exp(b)),
            "hr_lo": float(np.exp(b - 1.96 * s)), "hr_hi": float(np.exp(b + 1.96 * s)),
            "z": float(zz), "p": float(2 * stats.norm.sf(abs(zz))),
        } for n, b, s, zz in zip(self.names, self.beta, self.se, z)]

    def likelihood_ratio(self):
        chi2 = 2 * (self.loglik - self.loglik_null)
        return float(chi2), float(stats.chi2.sf(chi2, len(self.beta)))


def _sorted(time, event, X):
    order = np.argsort(-np.asarray(time, float), kind="stable")  # longest first, so risk sets are prefixes
    return np.asarray(time, float)[order], np.asarray(event, int)[order], np.asarray(X, float)[order]


def partial_likelihood(beta, time, event, X):
    """Log partial likelihood, score and information, Breslow ties.

    Expects rows sorted by decreasing time, so that the risk set of the i-th
    row is every row up to the last one sharing its time.
    """
    eta = X @ beta
    shift = eta.max()
    w = np.exp(eta - shift)
    s0 = np.cumsum(w)
    s1 = np.cumsum(w[:, None] * X, axis=0)
    s2 = np.cumsum(w[:, None, None] * X[:, :, None] * X[:, None, :], axis=0)
    # Everyone with the same time is in each other's risk set: use the last row of each tie block.
    last = np.searchsorted(-time, -time, side="right") - 1
    s0, s1, s2 = s0[last], s1[last], s2[last]

    ev = event == 1
    loglik = float(np.sum(eta[ev] - shift - np.log(s0[ev])))
    mean = s1[ev] / s0[ev, None]
    score = np.sum(X[ev] - mean, axis=0)
    info = np.sum(s2[ev] / s0[ev, None, None] - mean[:, :, None] * mean[:, None, :], axis=0)
    return loglik, score, info


def fit(time, event, X, names, tol=1e-9, max_iter=50):
    """Newton-Raphson with step halving if a step lowers the likelihood."""
    X = np.asarray(X, float)
    means = X.mean(axis=0)
    t, e, Xc = _sorted(time, event, X - means)
    beta = np.zeros(X.shape[1])
    loglik, score, info = partial_likelihood(beta, t, e, Xc)
    loglik_null = loglik
    for iteration in range(1, max_iter + 1):
        step = np.linalg.solve(info, score)
        for _ in range(30):
            new = beta + step
            new_loglik, new_score, new_info = partial_likelihood(new, t, e, Xc)
            if new_loglik >= loglik - 1e-12:
                break
            step = step / 2
        converged = abs(new_loglik - loglik) < tol
        beta, loglik, score, info = new, new_loglik, new_score, new_info
        if converged:
            break
    se = np.sqrt(np.diag(np.linalg.inv(info)))
    base_times, base_cumhaz = breslow_baseline(time, event, (X - means) @ beta)
    return CoxModel(list(names), beta, se, means, loglik, loglik_null, iteration, base_times, base_cumhaz)


def breslow_baseline(time, event, lp):
    """Breslow's baseline cumulative hazard at the mean covariates:
    H0(t) = sum over event times t_j <= t of d_j / sum over the risk set of exp(lp)."""
    time, event, risk = np.asarray(time, float), np.asarray(event, int), np.exp(lp)
    times = np.unique(time[event == 1])
    increments = np.array([((time == t) & (event == 1)).sum() / risk[time >= t].sum() for t in times])
    return times, np.cumsum(increments)


def schoenfeld_test(model, time, event, X):
    """Test of proportional hazards (Grambsch and Therneau, 1994).

    If an effect drifts over time, the scaled Schoenfeld residuals for that
    predictor trend with time. Each predictor is tested by regressing its
    scaled residuals on the rank of the event time; the global test combines
    them. Returns a p-value per predictor and overall.
    """
    Xc = np.asarray(X, float) - model.means
    t, e, Xs = _sorted(time, event, Xc)
    w = np.exp(Xs @ model.beta)
    s0 = np.cumsum(w)
    s1 = np.cumsum(w[:, None] * Xs, axis=0)
    last = np.searchsorted(-t, -t, side="right") - 1
    ev = e == 1
    resid = Xs[ev] - s1[last][ev] / s0[last][ev, None]
    d = ev.sum()
    var = np.linalg.inv(partial_likelihood(model.beta, t, e, Xs)[2])
    scaled = d * resid @ var + model.beta  # Grambsch-Therneau scaling
    g = stats.rankdata(t[ev])
    g = g - g.mean()
    # Score test per predictor: T = (g . r)^2 / (d * var_kk * sum g^2)
    num = (g @ (resid @ var * d)) ** 2
    per = num / (d * np.diag(var) * (g @ g))
    u = g @ (resid @ var * d)
    glob = float(u @ np.linalg.solve(d * var * (g @ g), u))
    return {
        "per_predictor": {n: float(stats.chi2.sf(x, 1)) for n, x in zip(model.names, per)},
        "global_chi2": glob, "global_df": len(model.names),
        "global_p": float(stats.chi2.sf(glob, len(model.names))),
        "times": t[ev].tolist(), "scaled": scaled.tolist(),
    }
