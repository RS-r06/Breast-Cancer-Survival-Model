"""How good are the predictions? Measures for survival data with censoring.

Some women leave follow-up before 5 years without an event (censored), so
their 5-year outcome is unknown. Simply dropping them biases every measure.
The measures here either use only pairs whose order is known (concordance)
or reweight the women with known outcomes by the inverse probability of
still being followed up (IPCW), which corrects for those lost.

Following the guidance in McLernon et al. (2023), Annals of Internal
Medicine: discrimination, calibration, overall accuracy and clinical use.
"""

import numpy as np

from .cox import fit
from .km import kaplan_meier, survival_at


def harrell_c(time, event, risk, horizon):
    """Share of usable pairs the model orders correctly.

    A pair is usable when one woman had an event before the horizon and
    before the other woman's time. It is concordant if she had the higher
    predicted risk. Tied predictions count half.
    """
    time, event, risk = map(np.asarray, (time, event, risk))
    conc = total = 0.0
    for i in np.flatnonzero((event == 1) & (time <= horizon)):
        later = time > time[i]
        total += later.sum()
        conc += (risk[i] > risk[later]).sum() + 0.5 * (risk[i] == risk[later]).sum()
    return conc / total


def censoring_survival(time, event):
    """G(t): Kaplan-Meier of the censoring times, the chance of still being followed up."""
    times, surv, _ = kaplan_meier(time, 1 - np.asarray(event))

    def G(t, left_limit=False):
        if len(times) == 0:  # nobody censored: follow-up is complete
            return np.ones_like(np.asarray(t, float))
        side = "left" if left_limit else "right"
        idx = np.searchsorted(times, t, side=side) - 1
        return np.where(idx < 0, 1.0, surv[np.clip(idx, 0, None)])

    return G


def ipcw_weights(time, event, horizon):
    """Weights for the known 5-year outcomes; 0 for women censored before the horizon.

    Cases had the event by the horizon. Controls were still event-free at
    the horizon, which includes women whose follow-up was cut off exactly
    there. The censoring curve is read just before each time (its left
    limit), so the cut-off at the horizon itself does not count as loss to
    follow-up.
    """
    time, event = np.asarray(time, float), np.asarray(event, int)
    G = censoring_survival(time, event)
    case = (time <= horizon) & (event == 1)
    control = time >= horizon
    control &= ~case
    w = np.zeros(len(time))
    w[case] = 1 / G(time[case], left_limit=True)
    w[control] = 1 / G(horizon, left_limit=True)
    return case, control, w


def auc_at(time, event, risk, horizon):
    """Time-dependent AUC at the horizon (cumulative cases, dynamic controls), IPCW."""
    case, control, w = ipcw_weights(time, event, horizon)
    risk = np.asarray(risk)
    rc, wc = risk[case], w[case]
    rn, wn = risk[control], w[control]
    num = sum(wi * (wn[(ri > rn)].sum() + 0.5 * wn[(ri == rn)].sum()) for ri, wi in zip(rc, wc))
    return num / (wc.sum() * wn.sum())


def brier_at(time, event, risk, horizon):
    """Brier score at the horizon, IPCW (Graf et al., 1999): mean squared
    difference between predicted risk and what happened."""
    case, control, w = ipcw_weights(time, event, horizon)
    risk = np.asarray(risk)
    return float(np.sum(w[case] * (1 - risk[case]) ** 2) + np.sum(w[control] * risk[control] ** 2)) / len(risk)


def scaled_brier(time, event, risk, horizon):
    """Index of prediction accuracy: 1 - Brier(model) / Brier(no-information model).

    The no-information model gives everyone the cohort's Kaplan-Meier risk.
    0 means no better than that; 1 is perfect.
    """
    s, _ = survival_at(time, event, horizon)
    null = np.full(len(risk), 1 - s)
    return 1 - brier_at(time, event, risk, horizon) / brier_at(time, event, null, horizon)


def calibration_in_the_large(time, event, risk, horizon):
    """Observed (Kaplan-Meier) risk against mean predicted risk, as an O/E ratio."""
    s, (lo, hi) = survival_at(time, event, horizon)
    expected = float(np.mean(risk))
    observed = 1 - s
    return {"observed": observed, "observed_ci": [1 - hi, 1 - lo], "expected": expected,
            "o_e": observed / expected, "o_e_ci": [(1 - hi) / expected, (1 - lo) / expected]}


def calibration_slope(time, event, lp):
    """Refit a Cox model with the linear predictor as its only input.

    A slope of 1 means the predicted spread of risk is right. Below 1 means
    predictions are too extreme (the usual sign of overfitting), above 1 too
    timid.
    """
    m = fit(time, event, np.asarray(lp)[:, None], ["lp"])
    return float(m.beta[0]), float(m.se[0])


def calibration_groups(time, event, risk, horizon, groups=10):
    """Observed Kaplan-Meier risk in each tenth of predicted risk."""
    order = np.argsort(risk)
    rows = []
    for chunk in np.array_split(order, groups):
        s, (lo, hi) = survival_at(np.asarray(time)[chunk], np.asarray(event)[chunk], horizon)
        rows.append({"predicted": float(np.mean(np.asarray(risk)[chunk])),
                     "observed": 1 - s, "lo": 1 - hi, "hi": 1 - lo, "n": len(chunk)})
    return rows


def net_benefit(time, event, risk, thresholds, horizon):
    """Decision curve analysis for a survival outcome (Vickers et al., 2008).

    Suppose women whose predicted 5-year risk is at least p_t get extra
    treatment. Net benefit counts true positives and charges each false
    positive the odds p_t / (1 - p_t): the trade-off a clinician accepts when
    choosing that threshold. Among the women treated, who goes on to have an
    event is estimated with Kaplan-Meier, so censoring is handled.
    """
    time, event, risk = map(np.asarray, (time, event, risk))
    n = len(time)
    s_all, _ = survival_at(time, event, horizon)
    rows = []
    for pt in thresholds:
        odds = pt / (1 - pt)
        treated = risk >= pt
        if treated.any():
            s, _ = survival_at(time[treated], event[treated], horizon)
            tp = (1 - s) * treated.sum() / n
            fp = s * treated.sum() / n
            model = tp - fp * odds
        else:
            model = 0.0
        treat_all = (1 - s_all) - s_all * odds
        rows.append({"threshold": float(pt), "model": float(model), "treat_all": float(treat_all), "treat_none": 0.0})
    return rows


def bootstrap(fn, n_patients, n_boot=1000, seed=0, confidence=0.95):
    """Percentile interval for any statistic of the patients, resampling patients."""
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_boot):
        idx = rng.integers(0, n_patients, n_patients)
        try:
            values.append(fn(idx))
        except (np.linalg.LinAlgError, ZeroDivisionError, ValueError):
            continue
    a = (1 - confidence) / 2
    return [float(np.quantile(values, a)), float(np.quantile(values, 1 - a))]


def optimism(time, event, X, names, horizon, n_boot=200, seed=0):
    """Harrell's bootstrap estimate of how flattering the development data are.

    Refit the model on a bootstrap sample, measure it on that sample and on
    the original data; the average gap is the optimism. Subtracting it from
    the apparent performance estimates performance on new data from the same
    population.
    """
    time, event, X = np.asarray(time), np.asarray(event), np.asarray(X)
    rng = np.random.default_rng(seed)
    gaps_c, slopes = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, len(time), len(time))
        m = fit(time[idx], event[idx], X[idx], names)
        c_boot = harrell_c(time[idx], event[idx], m.risk(X[idx], horizon), horizon)
        c_orig = harrell_c(time, event, m.risk(X, horizon), horizon)
        gaps_c.append(c_boot - c_orig)
        slopes.append(calibration_slope(time, event, m.linear_predictor(X))[0])
    return {"c_optimism": float(np.mean(gaps_c)), "slope_shrinkage": float(np.mean(slopes))}
