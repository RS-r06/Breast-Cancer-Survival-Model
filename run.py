"""Develop the model on Rotterdam, validate it on GBSG, write results and charts.

    python run.py
"""

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from bcsurv import data
from bcsurv.cox import fit, schoenfeld_test
from bcsurv.km import kaplan_meier, logrank, survival_at
from bcsurv.validate import (
    auc_at, bootstrap, brier_at, calibration_groups, calibration_in_the_large, calibration_slope,
    harrell_c, net_benefit, optimism, scaled_brier,
)

OUT = Path("results")
DOCS = Path("docs")
H = data.HORIZON
THRESHOLDS = np.round(np.arange(0.05, 0.71, 0.01), 2)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=130)
    plt.close(fig)
    print(f"  saved {OUT / name}")


def describe(frame):
    s, (lo, hi) = survival_at(frame["time"], frame["event"], H)
    return {
        "patients": int(len(frame)),
        "events_by_5y": int(frame["event"].sum()),
        "rfs_5y": s, "rfs_5y_ci": [lo, hi],
        "age_median": float(frame["age"].median()),
        "postmenopausal_pct": float(100 * frame["meno"].mean()),
        "tumour_over_20mm_pct": float(100 * (frame["size_20_50"] + frame["size_over_50"]).mean()),
        "grade3_pct": float(100 * frame["grade3"].mean()),
        "nodes_median": float(frame["nodes"].median()),
        "nodes_over_3_pct": float(100 * (frame["nodes"] > 3).mean()),
        "hormone_therapy_pct": float(100 * frame["hormon"].mean()),
    }


def performance(model, frame, n_boot=1000, seed=1):
    """Every validation measure, with bootstrap intervals over patients."""
    X = frame[data.PREDICTORS].to_numpy(float)
    t, e = frame["time"].to_numpy(), frame["event"].to_numpy()
    risk, lp = model.risk(X, H), model.linear_predictor(X)
    slope, slope_se = calibration_slope(t, e, lp)
    out = {
        "c_index": harrell_c(t, e, risk, H),
        "auc_5y": auc_at(t, e, risk, H),
        "brier_5y": brier_at(t, e, risk, H),
        "scaled_brier_5y": scaled_brier(t, e, risk, H),
        "calibration_slope": slope, "calibration_slope_ci": [slope - 1.96 * slope_se, slope + 1.96 * slope_se],
        **{f"cal_{k}": v for k, v in calibration_in_the_large(t, e, risk, H).items()},
    }
    for key, fn in {
        "c_index": lambda i: harrell_c(t[i], e[i], risk[i], H),
        "auc_5y": lambda i: auc_at(t[i], e[i], risk[i], H),
        "scaled_brier_5y": lambda i: scaled_brier(t[i], e[i], risk[i], H),
    }.items():
        out[f"{key}_ci"] = bootstrap(fn, len(t), n_boot=n_boot, seed=seed)
    return out, risk


def chart_cohorts(dev, val):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for frame, label, color in [(dev, "Rotterdam (development)", "#1f6fb2"), (val, "GBSG (validation)", "#c2410c")]:
        times, surv, _ = kaplan_meier(frame["time"], frame["event"])
        ax.step(np.r_[0, times], np.r_[1, surv], where="post", color=color, label=f"{label}, n = {len(frame)}")
    ax.set_xlim(0, H)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Years after surgery")
    ax.set_ylabel("Recurrence-free survival")
    ax.set_title("Kaplan-Meier: share of women alive without recurrence")
    ax.legend(loc="lower left")
    save(fig, "km_cohorts.png")


def chart_forest(model):
    rows = model.table()[::-1]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    y = np.arange(len(rows))
    ax.errorbar([r["hr"] for r in rows], y,
                xerr=[[r["hr"] - r["hr_lo"] for r in rows], [r["hr_hi"] - r["hr"] for r in rows]],
                fmt="s", color="#1f2937", capsize=3)
    ax.axvline(1, color="#9ca3af", ls="--")
    ax.set_yticks(y, [data.LABELS[r["name"]] for r in rows])
    ax.set_xscale("log")
    ticks = [0.5, 0.75, 1, 1.5, 2]
    ax.set_xticks(ticks, [str(t) for t in ticks])
    ax.minorticks_off()
    ax.set_xlabel("Hazard ratio with 95% CI (log scale). Right of 1: higher risk")
    ax.set_title("What drives risk in the model (Rotterdam, node-positive)")
    save(fig, "hazard_ratios.png")


def chart_calibration(groups, perf):
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    pred = [g["predicted"] for g in groups]
    obs = [g["observed"] for g in groups]
    ax.errorbar(pred, obs, yerr=[[o - g["lo"] for o, g in zip(obs, groups)], [g["hi"] - o for o, g in zip(obs, groups)]],
                fmt="o", color="#c2410c", capsize=3, label="GBSG tenths of predicted risk")
    ax.plot([0, 1], [0, 1], color="#9ca3af", ls="--", label="Perfect calibration")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted 5-year risk of recurrence or death")
    ax.set_ylabel("Observed (Kaplan-Meier) 5-year risk")
    ax.set_title(f"Calibration on GBSG: O/E {perf['cal_o_e']:.2f}, slope {perf['calibration_slope']:.2f}", fontsize=10)
    ax.legend(loc="upper left", fontsize=8)
    save(fig, "calibration.png")


def chart_decision(rows):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    th = [r["threshold"] for r in rows]
    ax.plot(th, [r["model"] for r in rows], color="#c2410c", lw=2, label="Treat if the model's risk is above the threshold")
    ax.plot(th, [r["treat_all"] for r in rows], color="#1f6fb2", label="Treat everyone")
    ax.plot(th, [0] * len(th), color="#1f2937", label="Treat no one")
    ax.set_ylim(-0.05, max(r["treat_all"] for r in rows) + 0.05)
    ax.set_xlabel("Risk threshold for extra treatment")
    ax.set_ylabel("Net benefit")
    ax.set_title("Decision curve on GBSG: when does using the model beat simple policies?")
    ax.legend(fontsize=8)
    save(fig, "decision_curve.png")


def chart_risk_groups(val, risk, cuts):
    group = np.digitize(risk, cuts)
    names = ["Low", "Middle", "High"]
    colors = ["#15803d", "#ca8a04", "#b91c1c"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for g in range(3):
        m = group == g
        times, surv, _ = kaplan_meier(val["time"][m], val["event"][m])
        ax.step(np.r_[0, times], np.r_[1, surv], where="post", color=colors[g],
                label=f"{names[g]} predicted risk, n = {m.sum()}")
    test = logrank(val["time"], val["event"], group)
    ax.set_xlim(0, H)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Years after surgery")
    ax.set_ylabel("Recurrence-free survival")
    ax.set_title(f"GBSG split by the model's risk groups (cut-offs from Rotterdam). Log-rank p = {test['p']:.1e}", fontsize=10)
    ax.legend(loc="lower left")
    save(fig, "risk_groups.png")
    return test


def chart_schoenfeld(model, ph, names=("log_er", "log_nodes")):
    """Scaled Schoenfeld residuals over time: a flat line means the effect is constant."""
    times = np.array(ph["times"])
    scaled = np.array(ph["scaled"])
    fig, axes = plt.subplots(1, len(names), figsize=(11, 4))
    edges = np.quantile(times, np.linspace(0, 1, 11))
    for ax, name in zip(axes, names):
        k = model.names.index(name)
        r = scaled[:, k]
        ax.plot(times, r, ".", color="#9ca3af", ms=3, alpha=0.5)
        mids, means, ses = [], [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (times >= lo) & (times <= hi)
            mids.append(times[m].mean())
            means.append(r[m].mean())
            ses.append(r[m].std(ddof=1) / np.sqrt(m.sum()))
        ax.errorbar(mids, means, yerr=1.96 * np.array(ses), fmt="o-", color="#c2410c", capsize=3,
                    label="Mean in each tenth of event times, 95% CI")
        ax.axhline(model.beta[k], color="#1f2937", ls="--", label="Model's constant coefficient")
        lim = np.quantile(np.abs(r - model.beta[k]), 0.98)
        ax.set_ylim(model.beta[k] - lim, model.beta[k] + lim)
        ax.set_xlabel("Years after surgery (event times)")
        ax.set_ylabel("Effect on log hazard")
        pv = ph["per_predictor"][name]
        ax.set_title(f"{data.LABELS[name]}: PH test p {'< 0.001' if pv < 0.001 else f'= {pv:.3f}'}", fontsize=10)
        ax.legend(fontsize=7, loc="upper right")
    fig.suptitle("Is each effect constant over time? (scaled Schoenfeld residuals)")
    save(fig, "proportional_hazards.png")


def examples(model):
    """A few profiles with the Python model's 5-year risk, for checking the browser calculator."""
    profiles = [
        {"age": 45, "meno": 0, "size_mm": 15, "grade": 2, "nodes": 1, "pgr": 200, "er": 150, "hormon": 1},
        {"age": 60, "meno": 1, "size_mm": 35, "grade": 3, "nodes": 6, "pgr": 10, "er": 40, "hormon": 0},
        {"age": 70, "meno": 1, "size_mm": 60, "grade": 3, "nodes": 20, "pgr": 0, "er": 0, "hormon": 0},
    ]
    out = []
    for p in profiles:
        x = [p["age"], p["meno"], int(20 < p["size_mm"] <= 50), int(p["size_mm"] > 50), int(p["grade"] == 3),
             np.log1p(p["nodes"]), np.log1p(p["pgr"]), np.log1p(p["er"]), p["hormon"]]
        out.append({"inputs": p, "risk_5y": float(model.risk(np.array([x]), H)[0])})
    return out


def export_calculator(model, perf, dev_desc):
    """Everything the browser calculator needs to reproduce the model's predictions."""
    grid = np.round(np.arange(0, H + 1e-9, 1 / 12), 4)
    s0 = model.survival(model.means[None, :], grid)  # baseline survival at the mean profile
    DOCS.mkdir(exist_ok=True)
    (DOCS / "model.json").write_text(json.dumps({
        "predictors": model.names, "beta": model.beta.tolist(), "means": model.means.tolist(),
        "times": grid.tolist(), "baseline_survival": s0.tolist(),
        "validation": {"c_index": perf["c_index"], "o_e": perf["cal_o_e"], "slope": perf["calibration_slope"]},
        "development_patients": dev_desc["patients"],
        "examples": examples(model),
    }))
    print(f"  saved {DOCS / 'model.json'}")


def main():
    start = time.time()
    OUT.mkdir(exist_ok=True)

    print("Cohorts")
    dev, val, dev_all = data.rotterdam(), data.gbsg(), data.rotterdam(node_positive_only=False)
    cohorts = {"rotterdam_node_positive": describe(dev), "gbsg": describe(val), "rotterdam_all": describe(dev_all)}
    for k, v in cohorts.items():
        print(f"  {k}: {v['patients']} women, {v['events_by_5y']} events by 5 years, 5-year RFS {v['rfs_5y']:.3f}")

    print("Model")
    X = dev[data.PREDICTORS].to_numpy(float)
    model = fit(dev["time"], dev["event"], X, data.PREDICTORS)
    lr, lr_p = model.likelihood_ratio()
    ph = schoenfeld_test(model, dev["time"], dev["event"], X)
    print(f"  converged in {model.iterations} iterations, likelihood ratio {lr:.1f} on 9 df (p = {lr_p:.1e})")
    print(f"  proportional hazards: global p = {ph['global_p']:.3f}; "
          + ", ".join(f"{k} {v:.3f}" for k, v in ph["per_predictor"].items()))
    for r in model.table():
        print(f"    {r['name']:<13} HR {r['hr']:.2f} ({r['hr_lo']:.2f} to {r['hr_hi']:.2f}) p = {r['p']:.2g}")

    print("Internal validation (Rotterdam, 200 bootstraps)")
    apparent, _ = performance(model, dev, n_boot=200)
    opt = optimism(dev["time"].to_numpy(), dev["event"].to_numpy(), X, data.PREDICTORS, H, n_boot=200)
    print(f"  apparent C {apparent['c_index']:.3f}, optimism {opt['c_optimism']:.4f}, "
          f"corrected C {apparent['c_index'] - opt['c_optimism']:.3f}, bootstrap shrinkage {opt['slope_shrinkage']:.3f}")

    print("External validation (GBSG, 1000 bootstraps)")
    perf, risk = performance(model, val)
    for k in ["c_index", "auc_5y", "scaled_brier_5y", "calibration_slope", "cal_o_e"]:
        ci = perf.get(f"{k}_ci") or perf.get(k.replace("cal_o_e", "cal_o_e_ci"))
        print(f"  {k}: {perf[k]:.3f} ({ci[0]:.3f} to {ci[1]:.3f})")
    print(f"  observed 5-year risk {perf['cal_observed']:.3f}, mean predicted {perf['cal_expected']:.3f}")
    groups = calibration_groups(val["time"].to_numpy(), val["event"].to_numpy(), risk, H)
    decision = net_benefit(val["time"].to_numpy(), val["event"].to_numpy(), risk, THRESHOLDS, H)
    useful = [r["threshold"] for r in decision if r["model"] > max(r["treat_all"], 0) + 1e-9]

    print("Sensitivity: model built on all Rotterdam patients")
    X_all = dev_all[data.PREDICTORS].to_numpy(float)
    model_all = fit(dev_all["time"], dev_all["event"], X_all, data.PREDICTORS)
    perf_all, _ = performance(model_all, val, n_boot=200)
    print(f"  GBSG C {perf_all['c_index']:.3f}, O/E {perf_all['cal_o_e']:.3f}, slope {perf_all['calibration_slope']:.3f}")

    print("Charts")
    chart_cohorts(dev, val)
    chart_forest(model)
    chart_schoenfeld(model, ph)
    chart_calibration(groups, perf)
    chart_decision(decision)
    dev_risk = model.risk(X, H)
    cuts = np.quantile(dev_risk, [1 / 3, 2 / 3])
    risk_test = chart_risk_groups(val, risk, cuts)
    export_calculator(model, perf, cohorts["rotterdam_node_positive"])

    (OUT / "metrics.json").write_text(json.dumps({
        "horizon_years": H, "cohorts": cohorts,
        "model": {"coefficients": model.table(), "likelihood_ratio": lr, "likelihood_ratio_p": lr_p,
                  "iterations": model.iterations,
                  "proportional_hazards": {k: v for k, v in ph.items() if k not in ("times", "scaled")}},
        "internal_validation": {"apparent": apparent, **opt,
                                "corrected_c_index": apparent["c_index"] - opt["c_optimism"]},
        "external_validation": perf, "calibration_groups": groups,
        "decision_curve": decision, "thresholds_where_model_helps": useful,
        "risk_groups": {"cutoffs": cuts.tolist(), "logrank": risk_test},
        "sensitivity_all_rotterdam": {k: perf_all[k] for k in ("c_index", "c_index_ci", "cal_o_e", "calibration_slope")},
    }, indent=1, default=float))
    print(f"Done in {time.time() - start:.0f} s")


if __name__ == "__main__":
    main()
