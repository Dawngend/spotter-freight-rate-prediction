"""Investigate quote transforms without opening the October holdout."""

import argparse
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

from eda import table
from features import FreightFeatures, distance_bucket


def relationships(frame, constant):
    rpm = frame.posted_rate / frame.distance
    return pd.Series({"rows": len(frame), "rank_correlation": rpm.corr(frame.quote_signal, method="spearman"),
                      "direct_median_error": (rpm - frame.quote_signal).abs().median(),
                      "mirror_median_error": (rpm + frame.quote_signal - constant).abs().median(),
                      "median_rpm_plus_quote": (rpm + frame.quote_signal).median()})


def regime_labels(frame):
    labels = {}
    for month, group in frame.groupby(frame.date.dt.month):
        corr = (group.posted_rate / group.distance).corr(group.quote_signal, method="spearman")
        labels[month] = "direct" if corr > .8 else "mirror" if corr < -.8 else "unresolved"
    return frame.date.dt.month.map(labels)


def predictors(fit, test, calendar):
    features = FreightFeatures(market=True).fit(fit)
    a, b = features.transform(fit), features.transform(test)
    # Quote itself is available at prediction time and is tested as a regime clue.
    a["quote_signal"], b["quote_signal"] = fit.quote_signal, test.quote_signal
    if not calendar:
        a = a.drop(columns=["month", "week_of_year"])
        b = b.drop(columns=["month", "week_of_year"])
    a, b = pd.get_dummies(a, dtype=float), pd.get_dummies(b, dtype=float)
    return a, b.reindex(columns=a.columns, fill_value=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args()
    data = pd.read_csv(args.data_dir / "train_test.csv", parse_dates=["date"])
    data = data[data.date < "2025-10-01"].copy()
    early = data[data.date < "2025-09-01"]
    labels = regime_labels(data)
    mirrored = early[regime_labels(early).eq("mirror")]
    constant = (mirrored.posted_rate / mirrored.distance + mirrored.quote_signal).median()
    sections = ["# Quote Signal Regimes",
                "Only January through September labels are used here. October is reserved for final model evaluation. No validation data is read or fitted.",
                "## Transform Check",
                "The sign in the task brief describes correlation with total rate. The RPM rank correlation is positive in direct months and negative in mirrored months.",
                f"The mirrored constant estimated from pre-September mirrored rows is {constant:.6f}. Compare RPM = quote_signal with RPM = constant - quote_signal. These are approximate relationships, not exact identities."]
    monthly = pd.DataFrame({str(month): relationships(group, constant)
                            for month, group in data.groupby(data.date.dt.to_period("M"))}).T
    sections.append(table(monthly))
    sections += ["## Finer Groups",
                 "Group summaries below pool the development months. The within-month table checks whether smaller groups reverse the dominant monthly relationship. Region cells use pickup and delivery latitude/longitude rounded down to five-degree bins. Weight and market bins are fixed before the analysis. Small groups below thirty rows are omitted from correlation summaries."]
    groups = {"week": data.date.dt.isocalendar().week.astype(str),
              "weekday": data.date.dt.dayofweek, "day_of_month": data.date.dt.day,
              "equipment": data.equipment, "distance": distance_bucket(data.distance),
              "weight": pd.cut(data.weight, [-np.inf, 0, 20000, 30000, 40000, np.inf]).astype(str),
              "market": pd.cut(data.market_index, [-np.inf, .9, 1, 1.1, 1.2, 1.3, np.inf]).astype(str)}
    for side in ["pickup", "delivery"]:
        groups[side + "_region"] = (np.floor(data[side + "_lat"] / 5).astype(int).astype(str)
                                     + ":" + np.floor(data[side + "_lon"] / 5).astype(int).astype(str))
    within = []
    for name, values in groups.items():
        pooled = pd.DataFrame({str(key): relationships(group, constant)
                               for key, group in data.groupby(values, observed=True) if len(group) >= 30}).T
        sections.extend([f"### {name.replace('_', ' ').title()}", table(pooled)])
        for month, monthly_data in data.groupby(data.date.dt.month):
            correlations = [relationships(g, constant).rank_correlation
                            for _, g in monthly_data.groupby(values.loc[monthly_data.index], observed=True) if len(g) >= 30]
            if correlations:
                expected = labels.loc[monthly_data.index].iloc[0]
                aligned = np.mean(np.array(correlations) > .8) if expected == "direct" else np.mean(np.array(correlations) < -.8) if expected == "mirror" else np.nan
                within.append({"group": name, "month": month, "cells": len(correlations),
                               "min_rank": min(correlations), "max_rank": max(correlations), "aligned_fraction": aligned})
    sections.extend(["### Within-Month Stability", table(pd.DataFrame(within)),
                     "## Predicting a Future Regime",
                     "Diagnostic labels are monthly direct, mirror or unresolved, assigned by RPM rank correlation thresholds of plus/minus 0.8. These target-derived labels are used only to evaluate regime prediction. Predictors contain available numeric features, equipment and quote, never posted_rate or RPM. Each forward window refits all preprocessing. Training accuracy measures memorization; future accuracy is the useful test. August introduces an unresolved class not present in earlier fitting months, which is itself a limitation of extrapolating this regime classifier."])
    results = []
    for calendar in [False, True]:
        for name in ["tree", "hist"]:
            for month in [6, 7, 8, 9]:
                fit = data[data.date.dt.month < month]
                test = data[data.date.dt.month == month]
                x, z = predictors(fit, test, calendar)
                model = (DecisionTreeClassifier(max_depth=4, min_samples_leaf=100, random_state=42)
                         if name == "tree" else HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=15, early_stopping=False, random_state=42))
                model.fit(x, labels.loc[fit.index])
                results.append({"model": name, "month_week": calendar, "test_month": month,
                                "train_accuracy": accuracy_score(labels.loc[fit.index], model.predict(x)),
                                "future_accuracy": accuracy_score(labels.loc[test.index], model.predict(z)),
                                "majority_accuracy": accuracy_score(labels.loc[test.index], np.repeat(labels.loc[fit.index].mode()[0], len(test)))})
    results = pd.DataFrame(results)
    sections += [table(results),
                 "## Conclusion",
                 "(b) The regime cannot be predicted reliably in future months from the tested validation-available columns, so quote_signal is excluded from the final model. Calendar splits can memorize observed regimes but do not supply a rule for unseen months. This is a failure to establish a usable rule, not proof that no rule can exist. August remains unresolved even within finer groups. The direct and mirrored near-target relationships also require source provenance before any deployment use."]
    output = Path(__file__).resolve().parents[1] / "reports"
    output.mkdir(exist_ok=True)
    (output / "quote_signal_regimes.md").write_text("\n\n".join(sections) + "\n", encoding="utf-8")
    print(results.to_string(index=False))
    print(f"Mirror Constant: {constant:.6f}")


if __name__ == "__main__":
    main()
