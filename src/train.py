"""Tune on September, freeze choices, evaluate October and write predictions."""

import argparse
import json
import os
from pathlib import Path

# One thread also avoids Windows restricted-environment thread-pool pipe errors.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from features import FreightFeatures, distance_bucket
from model_report import write_report


ROOT = Path(__file__).resolve().parents[1]


def metrics(actual, predicted):
    error = np.abs(np.asarray(actual) - predicted)
    percentage = error / np.asarray(actual) * 100
    return {"rows": len(actual), "MAE": mean_absolute_error(actual, predicted),
            "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
            "MAPE_pct": percentage.mean(), "MdAPE_pct": np.median(percentage),
            "R2": r2_score(actual, predicted) if len(actual) > 1 else np.nan}


def candidates(december=False):
    for model in ["median", "equipment_median"]:
        yield {"model": model, "target": "rpm", "market": False, "december": december, "parameter": 0}
    for market in ([False] if december else [False, True]):
        for target in ["rpm", "log_rate"]:
            for model, parameters in [("ridge", [1, 100]), ("hist_squared", [15, 31]), ("hist_absolute", [15, 31])]:
                for parameter in parameters:
                    yield {"model": model, "target": target, "market": market,
                           "december": december, "parameter": parameter}


def model_name(config):
    signal = "market" if config["market"] else "none"
    return f"{config['model']} / {config['target']} / {signal} / {config['parameter']}"


def fit_model(frame, config):
    features = FreightFeatures(config["market"], config["december"]).fit(frame)
    x = features.transform(frame)
    rpm = frame.posted_rate / x.distance_used
    if config["model"] == "median":
        estimator = float(rpm.median())
    elif config["model"] == "equipment_median":
        estimator = rpm.groupby(frame.equipment).median()
    else:
        numeric = x.columns.drop("equipment").tolist()
        encoder = ColumnTransformer([
            ("numeric", StandardScaler(), numeric),
            ("equipment", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["equipment"])])
        if config["model"] == "ridge":
            regressor = Ridge(alpha=config["parameter"])
        else:
            loss = "absolute_error" if config["model"] == "hist_absolute" else "squared_error"
            regressor = HistGradientBoostingRegressor(loss=loss, max_leaf_nodes=config["parameter"],
                                                      max_iter=200, learning_rate=.08,
                                                      min_samples_leaf=30, l2_regularization=1,
                                                      early_stopping=False, random_state=42)
        estimator = make_pipeline(encoder, regressor)
        target = rpm if config["target"] == "rpm" else np.log(frame.posted_rate)
        estimator.fit(x, target)
    return {"features": features, "estimator": estimator, "config": config}


def predict(model, frame, weekday=None):
    x = model["features"].transform(frame)
    if weekday is not None:
        x["day_of_week"] = weekday
    config, estimator = model["config"], model["estimator"]
    if config["model"] == "median":
        result = estimator * x.distance_used.to_numpy()
    elif config["model"] == "equipment_median":
        result = frame.equipment.map(estimator).to_numpy() * x.distance_used.to_numpy()
    else:
        result = estimator.predict(x)
        result = result * x.distance_used.to_numpy() if config["target"] == "rpm" else np.exp(result)
    # A fixed numerical floor, not a target-derived clipping threshold.
    return np.maximum(result, .01)


def tune(fit, september, december=False):
    results, winners = [], {}
    for config in candidates(december):
        model = fit_model(fit, config)
        scores = metrics(september.posted_rate, predict(model, september))
        row = {"candidate": model_name(config), **config, **scores}
        results.append(row)
        key = (config["model"], config["target"], config["market"])
        if key not in winners or scores["MAE"] < winners[key]["scores"]["MAE"]:
            winners[key] = {"model": model, "scores": scores}
        print(f"September {'December' if december else 'Validation'}: {model_name(config)} MAE={scores['MAE']:.2f}", flush=True)
    chosen = min(winners.values(), key=lambda item: (item["scores"]["MAE"], item["scores"]["RMSE"]))["model"]["config"]
    return pd.DataFrame(results), list(winners.values()), chosen


def slices(fit, test, predicted, features):
    known = pd.MultiIndex.from_frame(fit[["pickup", "delivery"]])
    seen = pd.MultiIndex.from_frame(test[["pickup", "delivery"]]).isin(known)
    groups = {"lane": pd.Series(np.where(seen, "seen", "unseen"), index=test.index),
              "equipment": test.equipment,
              "distance": distance_bucket(features.transform(test).distance_used)}
    records = []
    for category, values in groups.items():
        for name in values.dropna().unique():
            mask = values.eq(name).to_numpy()
            records.append({"slice": category, "group": str(name), **metrics(test.posted_rate[mask], predicted[mask])})
    return records


def evaluate(fit_august, fit_september, future, winners, chosen, purpose):
    october = future[future.date >= "2025-10-01"]
    results, slice_rows = [], []
    for item in winners:
        old_model = item["model"]
        config = old_model["config"]
        october_model = fit_model(fit_september, config)
        for label, fit, test, model in [("October (fit Jan-Sep)", fit_september, october, october_model),
                                        ("Sep-Oct (fit Jan-Aug)", fit_august, future, old_model)]:
            prediction = predict(model, test)
            results.append({"purpose": purpose, "window": label, "candidate": model_name(config),
                            "selected": config == chosen, **metrics(test.posted_rate, prediction)})
            for row in slices(fit, test, prediction, model["features"]):
                slice_rows.append({"purpose": purpose, "window": label, "candidate": model_name(config),
                                   "selected": config == chosen, **row})
    return results, slice_rows


def write_predictions(train, validation, chart, template, chosen, december):
    final = fit_model(train, chosen)
    chart_model = fit_model(train, december)
    predictions = pd.Series(predict(final, validation), index=validation.load_id)
    if template.load_id.duplicated().any() or set(template.load_id) != set(validation.load_id):
        raise ValueError("Template IDs do not match validation")
    output = template[["load_id"]].copy()
    output["predicted_rate"] = output.load_id.map(predictions)
    chart_output = chart.copy()
    chart_output["predicted_rate"] = predict(chart_model, chart)
    for values in [output.predicted_rate, chart_output.predicted_rate]:
        if not (np.isfinite(values).all() and (values > 0).all()):
            raise ValueError("Invalid predictions")
    output.to_csv(ROOT / "validation_predictions.csv", index=False, float_format="%.6f")
    chart_output.to_csv(ROOT / "outputs/december_chart_inputs_predicted.csv", index=False, float_format="%.6f")
    return final, chart_model, output, chart_output


def distance_policy_facts(train, train_x):
    """Numbers behind keeping the supplied distance for the loads whose road distance is above 3x the air miles."""
    flag = train_x.distance_flag.eq(1)
    short = train.distance.le(100)
    rpm = train.posted_rate / train.distance
    repaired_miles = (train.distance / train_x.distance_ratio.clip(lower=1e-9)) * 1.18  # old repair: 1.18 x air miles
    repaired_rate = rpm[short].median() * repaired_miles[flag]
    return {"flagged_rows": int(flag.sum()),
            "flagged_distances": sorted(train.distance[flag].unique().tolist()),
            "flagged_lanes": sorted({f"{a} to {b}" for a, b in zip(train.pickup[flag], train.delivery[flag])}),
            "min_distance_in_train": float(train.distance.min()),
            "rows_up_to_100_miles": int(short.sum()),
            "flagged_rate_min": float(train.posted_rate[flag].min()), "flagged_rate_median": float(train.posted_rate[flag].median()),
            "flagged_rate_max": float(train.posted_rate[flag].max()),
            "flagged_rpm_median": float(rpm[flag].median()), "short_rpm_median": float(rpm[short].median()),
            "old_repair_median_error_pct": float(100 * (repaired_rate - train.posted_rate[flag]).abs().div(train.posted_rate[flag]).median())}


def sanity(train, validation, final, output, chart, chart_model):
    train_x = final["features"].transform(train)
    x = final["features"].transform(validation)
    predicted = validation.load_id.map(output.set_index("load_id").predicted_rate)
    rpm = predicted / x.distance_used
    distribution = pd.DataFrame({"train_rpm": train.posted_rate / train.distance}).describe(percentiles=[.01, .05, .5, .95, .99])
    distribution["predicted_rpm"] = rpm.describe(percentiles=[.01, .05, .5, .95, .99])
    flagged = validation.loc[x.distance_flag.eq(1), ["load_id", "pickup", "delivery", "distance"]].copy()
    flagged["predicted_rate"] = predicted.loc[flagged.index]
    flagged["predicted_rpm"] = rpm.loc[flagged.index]
    distance_policy = distance_policy_facts(train, train_x)
    weekday_values = np.array([predict(chart_model, chart, weekday=day) for day in range(7)])
    known = pd.MultiIndex.from_frame(train[["pickup", "delivery"]])
    unseen = ~pd.MultiIndex.from_frame(validation[["pickup", "delivery"]]).isin(known)
    low, high = (train.posted_rate / train.distance).quantile([.001, .999])
    summary = {"train_rows": len(train), "validation_rows": len(validation), "chart_rows": len(chart),
               "missing_features": int(x.isna().sum().sum()), "missing_predictions": int(output.isna().sum().sum()),
               "train_flagged": int(train_x.distance_flag.sum()), "validation_flagged": int(x.distance_flag.sum()),
               "distance_policy": distance_policy,
               "rpm_screen_low": low, "rpm_screen_high": high,
               "rpm_outside_screen": int(((rpm < low) | (rpm > high)).sum()),
               "unseen_validation_lanes": int(unseen.sum()),
               "chart_min": chart.predicted_rate.min(), "chart_max": chart.predicted_rate.max(),
               "chart_range_pct": 100 * (chart.predicted_rate.max() - chart.predicted_rate.min()) / chart.predicted_rate.mean(),
               "weekday_max_effect": np.ptp(weekday_values, axis=0).max(),
               "chart_missing": int(chart.isna().sum().sum())}
    return summary, distribution, flagged


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args()
    (ROOT / "outputs").mkdir(exist_ok=True)
    (ROOT / "reports").mkdir(exist_ok=True)
    train = pd.read_csv(args.data_dir / "train_test.csv", parse_dates=["date"])
    august = train[train.date < "2025-09-01"]
    september = train[train.date.between("2025-09-01", "2025-09-30")]
    comparisons, winners, chosen = tune(august, september)
    december_comparisons, december_winners, december = tune(august, september, december=True)
    frozen = {"selection": "September MAE, RMSE tie-break; no October selection", "validation": chosen, "december": december}
    (ROOT / "outputs/model_choices.json").write_text(json.dumps(frozen, indent=2) + "\n", encoding="utf-8")
    print("Choices Frozen Before October Evaluation:", frozen, flush=True)
    fit_september = train[train.date < "2025-10-01"]
    future = train[train.date >= "2025-09-01"]
    results, slice_rows = evaluate(august, fit_september, future, winners, chosen, "validation")
    dec_results, dec_slices = evaluate(august, fit_september, future, december_winners, december, "december")
    results, slice_rows = pd.DataFrame(results + dec_results), pd.DataFrame(slice_rows + dec_slices)
    validation = pd.read_csv(args.data_dir / "validation.csv")
    chart = pd.read_csv(args.data_dir / "december_chart_inputs.csv")
    template = pd.read_csv(args.data_dir / "validation_predictions_template.csv")
    final, chart_model, output, chart_output = write_predictions(train, validation, chart, template, chosen, december)
    checks, distribution, flagged = sanity(train, validation, final, output, chart_output, chart_model)
    comparisons.to_csv(ROOT / "outputs/september_comparison.csv", index=False)
    december_comparisons.to_csv(ROOT / "outputs/december_september_comparison.csv", index=False)
    results.to_csv(ROOT / "outputs/holdout_metrics.csv", index=False)
    slice_rows.to_csv(ROOT / "outputs/holdout_slices.csv", index=False)
    write_report(comparisons, december_comparisons, results, slice_rows, chosen, december, checks, distribution, flagged)
    print(results[results.selected].to_string(index=False))
    print("Sanity Checks:", checks)


if __name__ == "__main__":
    main()
