"""Check the hand-written maths against established libraries and known answers."""

import numpy as np
import pandas as pd
import pytest
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test
from lifelines.utils import concordance_index
from statsmodels.duration.hazard_regression import PHReg

from bcsurv.cox import fit, schoenfeld_test
from bcsurv.km import kaplan_meier, logrank, survival_at
from bcsurv.validate import (
    auc_at, brier_at, calibration_in_the_large, calibration_slope, harrell_c, net_benefit, scaled_brier,
)


def simulate(n=600, beta=(0.7, -0.5), seed=0, censor_rate=0.15, tie_round=None):
    """Exponential survival with known hazard ratios and random censoring."""
    rng = np.random.default_rng(seed)
    X = np.column_stack([rng.normal(size=n), rng.binomial(1, 0.4, n)])
    rate = 0.2 * np.exp(X @ np.array(beta))
    t_event = rng.exponential(1 / rate)
    t_cens = rng.exponential(1 / censor_rate, n)
    time = np.minimum(t_event, t_cens)
    if tie_round:
        time = np.round(time, tie_round) + 1e-3
    event = (t_event <= t_cens).astype(int)
    return time, event, X


def test_kaplan_meier_matches_lifelines():
    time, event, _ = simulate(seed=1)
    times, surv, var = kaplan_meier(time, event)
    kmf = KaplanMeierFitter().fit(time, event)
    ref = kmf.survival_function_at_times(times).to_numpy()
    assert np.allclose(surv, ref)
    s, (lo, hi) = survival_at(time, event, 3.0)
    ci = kmf.confidence_interval_survival_function_
    row = ci[ci.index <= 3.0].iloc[-1]
    assert s == pytest.approx(float(kmf.predict(3.0)))
    assert lo == pytest.approx(row.iloc[0], abs=1e-6) and hi == pytest.approx(row.iloc[1], abs=1e-6)


def test_kaplan_meier_without_censoring_is_the_empirical_survival():
    t = np.array([1.0, 2, 2, 3, 5])
    times, surv, _ = kaplan_meier(t, np.ones(5, int))
    assert times.tolist() == [1, 2, 3, 5]
    assert np.allclose(surv, [0.8, 0.4, 0.2, 0.0])


def test_logrank_matches_lifelines_for_two_and_three_groups():
    time, event, X = simulate(seed=2)
    g2 = X[:, 1].astype(int)
    mine = logrank(time, event, g2)
    ref = logrank_test(time[g2 == 0], time[g2 == 1], event[g2 == 0], event[g2 == 1])
    assert mine["chi2"] == pytest.approx(ref.test_statistic, rel=1e-8)
    g3 = np.digitize(X[:, 0], [-0.5, 0.5])
    ref3 = multivariate_logrank_test(time, g3, event)
    assert logrank(time, event, g3)["chi2"] == pytest.approx(ref3.test_statistic, rel=1e-8)


def test_cox_recovers_known_hazard_ratios():
    time, event, X = simulate(n=4000, seed=3)
    m = fit(time, event, X, ["x", "group"])
    assert m.beta[0] == pytest.approx(0.7, abs=0.06)
    assert m.beta[1] == pytest.approx(-0.5, abs=0.08)


def test_cox_matches_statsmodels_with_tied_times():
    time, event, X = simulate(n=800, seed=4, tie_round=1)  # many tied times
    assert len(np.unique(time)) < len(time) / 3
    m = fit(time, event, X, ["x", "group"])
    ref = PHReg(time, X, status=event, ties="breslow").fit()
    assert np.allclose(m.beta, ref.params, atol=1e-6)
    assert np.allclose(m.se, ref.bse, rtol=1e-5)
    assert m.loglik == pytest.approx(ref.llf, rel=1e-8)


def test_baseline_survival_matches_lifelines_breslow():
    from lifelines import CoxPHFitter
    time, event, X = simulate(n=1500, seed=5)
    m = fit(time, event, X, ["x", "group"])
    frame = pd.DataFrame(X, columns=["x", "group"]).assign(T=time, E=event)
    ref = CoxPHFitter(baseline_estimation_method="breslow").fit(frame, "T", "E")
    # Unique times, so Breslow and lifelines' Efron ties agree.
    assert np.allclose(m.beta, ref.params_.to_numpy(), atol=1e-5)
    pred = ref.predict_survival_function(frame.iloc[:5][["x", "group"]], times=[2.0]).to_numpy().ravel()
    assert np.allclose(m.survival(X[:5], 2.0), pred, atol=1e-4)


def test_proportional_hazards_test_passes_when_true_and_fails_when_not():
    time, event, X = simulate(n=3000, seed=6)
    m = fit(time, event, X, ["x", "group"])
    assert schoenfeld_test(m, time, event, X)["global_p"] > 0.01
    # Make the group effect fade with time: hazard ratio exp(1.5) early, 1 late.
    rng = np.random.default_rng(7)
    g = rng.binomial(1, 0.5, 3000)
    early = rng.exponential(1 / (0.3 * np.exp(1.5 * g)))
    late = 2 + rng.exponential(1 / 0.3, 3000)
    t = np.where(early < 2, early, late)
    e = np.ones(3000, int)
    m2 = fit(t, e, g[:, None], ["group"])
    assert schoenfeld_test(m2, t, e, g[:, None])["global_p"] < 1e-6


def test_proportional_hazards_test_close_to_lifelines():
    from lifelines import CoxPHFitter
    from lifelines.statistics import proportional_hazard_test
    time, event, X = simulate(n=1000, seed=8)
    m = fit(time, event, X, ["x", "group"])
    frame = pd.DataFrame(X, columns=["x", "group"]).assign(T=time, E=event)
    cph = CoxPHFitter().fit(frame, "T", "E")
    ref = proportional_hazard_test(cph, frame, time_transform="rank")
    mine = schoenfeld_test(m, time, event, X)["per_predictor"]
    for name in ["x", "group"]:
        assert mine[name] == pytest.approx(float(ref.summary.loc[name, "p"]), abs=0.02)


def test_harrell_c_matches_lifelines_when_nothing_is_truncated():
    time, event, X = simulate(seed=9)
    risk = X @ np.array([0.7, -0.5])
    mine = harrell_c(time, event, risk, horizon=np.inf)
    assert mine == pytest.approx(concordance_index(time, -risk, event), abs=1e-9)


def test_perfect_and_useless_predictions():
    time, event, X = simulate(n=1500, seed=10, censor_rate=0.0001)
    truth = 1 - np.exp(-0.2 * np.exp(X @ np.array([0.7, -0.5])) * 3.0)  # true 3-year risk
    assert auc_at(time, event, truth, 3.0) > 0.7
    flat = np.full(len(time), 0.5)
    assert auc_at(time, event, flat, 3.0) == pytest.approx(0.5)
    assert scaled_brier(time, event, truth, 3.0) > 0.05
    observed = 1 - survival_at(time, event, 3.0)[0]
    # The truth is well calibrated: O/E close to 1, slope close to 1.
    assert calibration_in_the_large(time, event, truth, 3.0)["o_e"] == pytest.approx(1.0, abs=0.08)
    slope, _ = calibration_slope(time, event, X @ np.array([0.7, -0.5]))
    assert slope == pytest.approx(1.0, abs=0.1)
    # Predicting everyone's risk as the observed average beats nothing and nothing beats it.
    assert scaled_brier(time, event, np.full(len(time), observed), 3.0) == pytest.approx(0.0, abs=1e-9)


def test_brier_without_censoring_is_the_plain_mean_squared_error():
    time = np.array([1.0, 2.0, 4.0, 6.0])
    event = np.ones(4, int)
    risk = np.array([0.9, 0.6, 0.3, 0.1])
    outcome = (time <= 3).astype(float)
    assert brier_at(time, event, risk, 3.0) == pytest.approx(np.mean((risk - outcome) ** 2))


def test_net_benefit_of_treating_everyone():
    time, event, X = simulate(seed=11)
    rows = net_benefit(time, event, np.ones(len(time)), [0.1, 0.3], horizon=3.0)
    for r in rows:
        assert r["model"] == pytest.approx(r["treat_all"])  # a model that flags everyone is "treat all"
    none = net_benefit(time, event, np.zeros(len(time)), [0.1], horizon=3.0)
    assert none[0]["model"] == 0.0


def test_measures_work_when_follow_up_is_cut_off_at_the_horizon():
    # The study cuts follow-up at 5 years, so everyone still well has time
    # exactly 5.0 and no event. They are the controls and must be counted.
    from bcsurv.data import truncate
    time, event, X = simulate(n=1500, seed=12)
    t5, e5 = truncate(time, event, horizon=5.0)
    assert (t5 == 5.0).sum() > 100
    risk = 1 - np.exp(-0.2 * np.exp(X @ np.array([0.7, -0.5])) * 5.0)
    auc_cut = auc_at(t5, e5, risk, 5.0)
    assert np.isfinite(auc_cut) and auc_cut > 0.65
    # Cutting off after the horizon changes nothing about a 5-year measure.
    assert auc_cut == pytest.approx(auc_at(time, event, risk, 5.0), abs=1e-9)
    assert brier_at(t5, e5, risk, 5.0) == pytest.approx(brier_at(time, event, risk, 5.0), abs=1e-9)
    assert scaled_brier(t5, e5, risk, 5.0) > 0
