"""Read private freight data and write aggregate findings and four figures."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def table(frame):
    """Format a small summary without requiring tabulate."""
    frame = frame.reset_index()
    rows = [list(frame.columns)] + frame.astype(object).values.tolist()
    lines = []
    for i, row in enumerate(rows):
        cells = [
            (f"{v:,.0f}" if v.is_integer() else f"{v:,.4f}")
            if isinstance(v, float) else str(v)
            for v in row
        ]
        lines.append("| " + " | ".join(cells) + " |")
        if i == 0:
            lines.append("| " + " | ".join(["---"] * len(row)) + " |")
    return "\n".join(lines)


def prepare(frame):
    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["month"] = frame.date.dt.to_period("M").astype(str)
    coords = np.radians(frame[["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"]])
    lat1, lon1, lat2, lon2 = coords.to_numpy().T
    h = np.sin((lat2 - lat1) / 2) ** 2
    h += np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    frame["air_miles"] = 3958.7613 * 2 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))
    frame["distance_ratio"] = frame.distance / frame.air_miles.replace(0, np.nan)
    if "posted_rate" in frame:
        frame["rate_per_mile"] = frame.posted_rate / frame.distance.where(frame.distance > 0)
    return frame


def cities(frame):
    return pd.concat([
        frame[[side, f"{side}_lat", f"{side}_lon"]].set_axis(["city", "lat", "lon"], axis=1)
        for side in ["pickup", "delivery"]
    ], ignore_index=True)


def quality(frame, numeric_columns):
    city = cities(frame)
    unique = city.drop_duplicates()
    normalized = city.city.str.strip().str.casefold()
    lat = frame[["pickup_lat", "delivery_lat"]]
    lon = frame[["pickup_lon", "delivery_lon"]]
    return pd.Series({
        "Invalid dates": int(frame.date.isna().sum()),
        "Nonfinite numeric cells (excluding missing)": int(np.isinf(frame[numeric_columns]).sum().sum()),
        "Missing weight": int(frame.weight.isna().sum()),
        "Negative weight": int((frame.weight < 0).sum()),
        "Zero weight": int((frame.weight == 0).sum()),
        "Missing market_index": int(frame.market_index.isna().sum()),
        "Nonpositive distance": int((frame.distance <= 0).sum()),
        "Invalid latitude/longitude rows": int(((lat.abs() > 90).any(axis=1) | (lon.abs() > 180).any(axis=1)).sum()),
        "Zero straight-line distance": int(frame.air_miles.eq(0).sum()),
        "Distance below straight-line miles": int((frame.distance < frame.air_miles).sum()),
        "Distance above 3x straight-line miles": int((frame.distance_ratio > 3).sum()),
        "Cities": city.city.nunique(),
        "Cities with multiple coordinate pairs": int((unique.groupby("city").size() > 1).sum()),
        "Coordinate pairs with multiple names": int((unique.groupby(["lat", "lon"]).size() > 1).sum()),
        "Extra names differing only in case/space": city.city.nunique() - normalized.nunique(),
    })


def coverage(train, validation, columns):
    known = pd.MultiIndex.from_frame(train[columns].drop_duplicates())
    values = pd.MultiIndex.from_frame(validation[columns])
    unique = values.unique()
    return {"Train unique": len(known), "Validation unique": len(unique),
            "Seen unique": int(unique.isin(known).sum()),
            "Unseen unique": int((~unique.isin(known)).sum()),
            "Seen rows": int(values.isin(known).sum()),
            "Unseen rows": int((~values.isin(known)).sum())}


def signal_correlations(frame):
    return pd.Series({
        "Rows": len(frame),
        "Market vs rate Pearson": frame.market_index.corr(frame.posted_rate),
        "Market vs RPM Pearson": frame.market_index.corr(frame.rate_per_mile),
        "Market vs RPM Spearman": frame.market_index.corr(frame.rate_per_mile, method="spearman"),
        "Quote vs rate Pearson": frame.quote_signal.corr(frame.posted_rate),
        "Quote vs RPM Pearson": frame.quote_signal.corr(frame.rate_per_mile),
        "Quote vs RPM Spearman": frame.quote_signal.corr(frame.rate_per_mile, method="spearman"),
        "Median abs quote minus RPM": (frame.quote_signal - frame.rate_per_mile).abs().median(),
    })


def distribution(frame, columns):
    return frame[columns].describe(percentiles=[.01, .05, .25, .5, .75, .95, .99]).T


def monthly_correlations(train):
    columns = ["market_index", "quote_signal", "posted_rate", "rate_per_mile"]
    return train.groupby("month")[columns].apply(signal_correlations)


def save_charts(train, validation, output):
    output.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, col, label in zip(axes, ["posted_rate", "rate_per_mile"], ["Posted Rate", "Rate per Mile"]):
        ax.hist(train[col], bins=80, color="#176b87")
        ax.set(title=f"Training {label}", xlabel=label, ylabel="Rows (Log Scale)")
        ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(output / "target_distribution.png", dpi=150)
    plt.close(fig)

    monthly = train.groupby("month")[["posted_rate", "rate_per_mile"]].median()
    counts = pd.concat([train.month.value_counts().rename("Train"), validation.month.value_counts().rename("Validation")], axis=1).fillna(0).sort_index()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    counts.plot.bar(ax=axes[0], color=["#176b87", "#df8540"])
    axes[0].set(title="Rows by Month and Dataset", xlabel="Month", ylabel="Rows")
    monthly.rate_per_mile.plot(marker="o", ax=axes[1], color="#176b87")
    axes[1].set(title="Training Median Rate per Mile", xlabel="Month", ylabel="Rate per Mile")
    axes[1].tick_params(axis="x", rotation=90)
    fig.tight_layout()
    fig.savefig(output / "monthly_coverage.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    bins = axes[0].hexbin(train.distance, train.posted_rate, gridsize=45, bins="log", mincnt=1, cmap="viridis")
    fig.colorbar(bins, ax=axes[0], label="Rows per Hexagon")
    axes[0].set(title="Training Distance and Rate (Log Count Color)", xlabel="Distance (Miles)", ylabel="Posted Rate")
    axes[1].hist(train.distance_ratio, bins=np.linspace(1, 3, 60), alpha=.6, density=True, label="Train")
    axes[1].hist(validation.distance_ratio, bins=np.linspace(1, 3, 60), alpha=.6, density=True, label="Validation")
    axes[1].set(title="Distance / Straight-Line Miles (Range 1 to 3)", xlabel="Ratio (Values Above 3 Reported in Text)", ylabel="Density within Range")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output / "distance_checks.png", dpi=150)
    plt.close(fig)

    corr = monthly_correlations(train)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    corr[["Quote vs RPM Spearman", "Market vs RPM Spearman"]].plot(marker="o", ax=axes[0])
    axes[0].axhline(0, color="gray", linewidth=.7)
    axes[0].set(title="Signal Correlation by Training Month", xlabel="Month", ylabel="Spearman Correlation", ylim=(-1.05, 1.05))
    axes[0].tick_params(axis="x", rotation=90)
    axes[0].set_xticks(range(len(corr)), corr.index)
    axes[0].legend(loc="lower center", bbox_to_anchor=(.5, 1.12), fontsize=8)
    for label, frame in [("Train", train), ("Validation", validation)]:
        axes[1].hist(frame.market_index.dropna(), bins=np.linspace(.65, 1.5, 45), density=True, alpha=.5, label=label)
    axes[1].set(title="Market Index Distribution", xlabel="Market Index", ylabel="Density")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output / "signal_stability.png", dpi=150)
    plt.close(fig)


def completeness_sections(raw_train, raw_validation, train, validation):
    sections = ["# Freight Rate EDA Findings", "Computed from the three supplied CSV files. Source files are read only. Tables describe the raw data unless a subset is named. RPM means posted_rate divided by positive distance. Correlations use available pairs. Rates use the source's units; distances are treated as miles for the coordinate check."]
    sections += ["## Shape and Completeness"]
    for label, raw, frame in [("Train", raw_train, train), ("Validation", raw_validation, validation)]:
        sections.append(f"{label} has {len(raw):,} rows and {len(raw.columns)} columns, {raw.duplicated().sum():,} duplicate full rows and {raw.load_id.duplicated().sum():,} duplicate load_ids (extra occurrences). Dates span {frame.date.min():%Y-%m-%d} to {frame.date.max():%Y-%m-%d}.")
    schema = pd.DataFrame({"Train dtype": raw_train.dtypes.astype(str), "Train missing": raw_train.isna().sum().astype(str), "Validation dtype": raw_validation.dtypes.astype(str), "Validation missing": raw_validation.isna().sum().astype(str)}).fillna("absent")
    sections += [table(schema), f"Shared load_ids between datasets: {len(set(train.load_id) & set(validation.load_id)):,}. Dates are object strings in the CSV and parsed for this analysis. Validation has no posted_rate column, so its prediction error and target correlations cannot be measured."]
    return sections


def time_sections(train, validation):
    sections = ["## Time Coverage and Validation"]
    monthly_counts = pd.concat([train.month.value_counts().rename("Train rows"), validation.month.value_counts().rename("Validation rows")], axis=1).fillna(0).astype(int).sort_index()
    sections.append(table(monthly_counts))
    fit = train.date < "2025-09-01"
    tune = train.date.between("2025-09-01", "2025-09-30")
    final = train.date >= "2025-10-01"
    sections.append(f"Use dates through 2025-08-31 for fitting ({fit.sum():,} rows), 2025-09-01 to 2025-09-30 for tuning ({tune.sum():,}), and 2025-10-01 to 2025-10-31 as the untouched final holdout ({final.sum():,}). Also test a two-month forecast: fit through 2025-08-31 and evaluate September plus October ({(tune | final).sum():,} rows), matching the two future months in the supplied validation set. After selection, refit through 2025-10-31. Fit imputers, clipping thresholds and encoders inside each fitting window.")
    random_fit, random_test = train_test_split(train, test_size=.2, random_state=42)
    random_lanes = coverage(random_fit, random_test, ["pickup", "delivery"])
    temporal_lanes = coverage(train[fit], train[tune | final], ["pickup", "delivery"])
    sections.append(f"A seeded 80/20 random split gives {len(random_fit):,}/{len(random_test):,} rows, with {random_fit.month.nunique()} months in fitting and {random_test.month.nunique()} in testing. It puts {(random_fit.date >= '2025-09-01').sum():,} September/October rows into fitting. Only {random_lanes['Unseen rows']:,} random-test rows have unseen directed lanes, versus {temporal_lanes['Unseen rows']:,} in the two-month time holdout. This lets the model learn the same months and signal regimes it is tested on. It risks optimistic future performance; no model-score improvement is claimed by this EDA.")
    sections.append("The time holdout has fewer unseen-lane rows than the random holdout in this sample, so lane novelty is not the reason to prefer it. The future-month boundary is. Freeze model choices before examining October or the combined September/October result.")
    return sections + ["![Monthly Coverage](figures/monthly_coverage.png)"]


def target_sections(train):
    sections = ["## Target and Feature Relationships", table(distribution(train, ["posted_rate", "rate_per_mile"]))]
    outlier_counts = {}
    for col in ["posted_rate", "rate_per_mile"]:
        q1, q3 = train[col].quantile([.25, .75])
        low, high = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
        outlier_counts[col] = int(((train[col] < low) | (train[col] > high)).sum())
        sections.append(f"For {col}, the 1.5-IQR fences are {low:,.4f} and {high:,.4f}; {outlier_counts[col]:,} rows fall outside them. These are review flags, not proof of bad labels.")
    sections.append(f"Target zeros: {train.posted_rate.eq(0).sum():,}; negative targets: {(train.posted_rate < 0).sum():,}; skewness: {train.posted_rate.skew():.4f}. RPM zeros: {train.rate_per_mile.eq(0).sum():,}; negative RPM: {(train.rate_per_mile < 0).sum():,}.")
    corr = pd.DataFrame({f"{target} {method}": train[["distance", "weight", "market_index", "quote_signal"]].corrwith(train[target], method=method) for target in ["posted_rate", "rate_per_mile"] for method in ["pearson", "spearman"]})
    sections += [table(corr), "These are marginal associations, not causal effects. Weight correlations above include recorded negative weights."]
    valid_weight = train.weight > 0
    sections.append(f"Using only {valid_weight.sum():,} positive-weight rows, weight Pearson correlation is {train.loc[valid_weight, 'weight'].corr(train.loc[valid_weight, 'posted_rate']):.4f} with rate and {train.loc[valid_weight, 'weight'].corr(train.loc[valid_weight, 'rate_per_mile']):.4f} with RPM.")
    for col in ["equipment", "month"]:
        summary = train.groupby(col).agg(rows=("posted_rate", "size"), mean_rate=("posted_rate", "mean"), median_rate=("posted_rate", "median"), mean_rpm=("rate_per_mile", "mean"), median_rpm=("rate_per_mile", "median"))
        sections.append(table(summary))
    medians = train.groupby("equipment").rate_per_mile.median()
    sections.append(f"Distance is strongly associated with total rate (Pearson {train.distance.corr(train.posted_rate):.4f}), while longer loads have lower RPM (Spearman {train.distance.corr(train.rate_per_mile, method='spearman'):.4f}). Median RPM is {medians['Dry Van']:.4f} for Dry Van, {medians['Flatbed']:.4f} for Flatbed and {medians['Reefer']:.4f} for Reefer. Equipment comparisons are not adjusted for lane or month.")
    return sections + ["![Target Distribution](figures/target_distribution.png)"], outlier_counts


def signal_sections(train, validation):
    sections = ["## Signal Strength and Stability", table(pd.DataFrame({"All train": signal_correlations(train)}).T)]
    by_month = monthly_correlations(train)
    sections.append(table(by_month))
    median_rpm = train.rate_per_mile.median()
    near_copy = by_month[by_month['Median abs quote minus RPM'] < .01 * median_rpm]
    gaps = near_copy['Median abs quote minus RPM']
    ranks = near_copy['Quote vs RPM Spearman']
    mirrored = by_month[by_month['Quote vs RPM Spearman'] < -.9]
    mirrored_gaps = mirrored['Median abs quote minus RPM']
    sections.append(f"In {len(near_copy)} of {len(by_month)} training months ({', '.join(near_copy.index)}), median absolute quote_signal minus RPM is {gaps.min():.4f} to {gaps.max():.4f}, or {100 * gaps.min() / median_rpm:.2f}% to {100 * gaps.max() / median_rpm:.2f}% of the overall median RPM ({median_rpm:.4f}). Monthly Spearman correlations in those months are {ranks.min():.4f} to {ranks.max():.4f}. This near-copy relationship is strong evidence that quote_signal is target-derived in those months, rather than merely correlated with RPM. This is an inference from the values, not verified feature provenance. Treat quote_signal as a leak unless provenance proves it predates the quote; scores using it are not credible evidence of deployable prediction quality without that proof.")
    sections.append(f"The {len(mirrored)} strongly negative months ({', '.join(mirrored.index)}) have median absolute gaps of {mirrored_gaps.min():.4f} to {mirrored_gaps.max():.4f} and Spearman correlations of {mirrored['Quote vs RPM Spearman'].min():.4f} to {mirrored['Quote vs RPM Spearman'].max():.4f}, consistent with a mirrored target transform. August Spearman is {by_month.loc['2025-08', 'Quote vs RPM Spearman']:.4f}, consistent with noise. Market-to-RPM monthly Pearson correlation ranges from {by_month['Market vs RPM Pearson'].min():.4f} to {by_month['Market vs RPM Pearson'].max():.4f}. Report the signal-free model as the headline and require pre-quote provenance for either signal before considering it for deployment.")
    signal_dist = pd.concat({"Train": distribution(train, ["market_index", "quote_signal"]), "Validation": distribution(validation, ["market_index", "quote_signal"])}, names=["Dataset", "Feature"])
    sections.append(table(signal_dist))
    monthly_signals = pd.concat([train.assign(dataset="Train"), validation.assign(dataset="Validation")]).groupby(["dataset", "month"])[["market_index", "quote_signal"]].mean()
    sections.append(table(monthly_signals))
    for col in ["market_index", "quote_signal"]:
        shift = (validation[col].mean() - train[col].mean()) / train[col].std()
        std_ratio = validation[col].std() / train[col].std()
        outside = (validation[col] < train[col].min()) | (validation[col] > train[col].max())
        if abs(shift) > .5 or not .5 <= std_ratio <= 2:
            interpretation = "The distribution differs materially. Staying inside the training min/max does not rule out a shift in its center or spread."
        else:
            interpretation = "The mean is similar, but the spread is lower. Similar marginal values do not establish stable target relationships, since validation has no labels."
        sections.append(f"{col}: validation mean shift is {shift:.4f} training standard deviations; validation/train standard deviation ratio is {std_ratio:.4f}. There are {outside.sum():,} observed validation values outside the training min/max. Missingness is {train[col].isna().mean():.2%} in train and {validation[col].isna().mean():.2%} in validation. {interpretation}")
    sections.append(f"Market index is not a very strong marginal signal in this data: pooled RPM Pearson is {train.market_index.corr(train.rate_per_mile):.4f}. Quote is very strong within most months by rank, but August Spearman is {by_month.loc['2025-08', 'Quote vs RPM Spearman']:.4f}. A single stable quote-to-price rule is not supported by these month-level results.")
    return sections + ["![Signal Stability](figures/signal_stability.png)"]


def quality_sections(train, validation, outlier_counts, numeric_columns):
    sections = ["## Data Quality and Suggested Handling", table(pd.DataFrame({"Train": quality(train, numeric_columns), "Validation": quality(validation, numeric_columns)}))]
    sections.append(f"The nonfinite check scans the same {len(numeric_columns)} shared raw numeric columns in each dataset: {', '.join(numeric_columns)}. Missing values are counted separately; the target and derived columns are excluded from this comparison.")
    all_cities = cities(pd.concat([train, validation]))
    conflicts = (all_cities.drop_duplicates().groupby("city").size() > 1).sum()
    sections.append(f"Across both files, {conflicts:,} city names map to multiple coordinate pairs. The check finds internal consistency only; it does not establish that coordinates match real cities. Case/space checks cannot detect arbitrary misspellings. No external geocoding was used.")
    sections.append(table(pd.DataFrame({"Train distance / air miles": train.distance_ratio.describe(), "Validation distance / air miles": validation.distance_ratio.describe()})))
    sections += ["Straight-line distance uses the haversine formula with Earth radius 3,958.7613 miles. A road distance above three times straight-line distance is a screening threshold, not an impossible route. Check routing and units before replacing it.", "![Distance Checks](figures/distance_checks.png)"]
    issues = [
        ("Missing weight", int(train.weight.isna().sum()), int(validation.weight.isna().sum()), "Flag and impute from fitting-window equipment medians, with a global fallback."),
        ("Negative weight", int((train.weight < 0).sum()), int((validation.weight < 0).sum()), "Flag, set to missing in model input and impute. Do not assume absolute value is correct."),
        ("Missing market_index", int(train.market_index.isna().sum()), int(validation.market_index.isna().sum()), "Flag and impute using fitting-window values, or exclude in the signal-free model."),
        ("Distance above 3x air miles", int((train.distance_ratio > 3).sum()), int((validation.distance_ratio > 3).sum()), "Flag for route review. Keep supplied distance until a correction is verified."),
        ("Rate outside 1.5-IQR fences", outlier_counts["posted_rate"], "No target", "Flag and inspect with lane, equipment and RPM. Compare robust loss or log target; do not drop by price alone."),
        ("RPM outside 1.5-IQR fences", outlier_counts["rate_per_mile"], "No target", "Flag and compare sensitivity. If clipping helps a time holdout, learn bounds within its fitting window."),
    ]
    sections.append(table(pd.DataFrame(issues, columns=["Issue", "Train rows", "Validation rows", "Suggested Handling"]).set_index("Issue")))
    sections.append("Issue counts can overlap. Keep all validation rows for final prediction. No source values were changed, no rows were dropped and no clipping was applied in this EDA.")
    return sections


def coverage_sections(train, validation):
    sections = ["## Lane and Equipment Coverage"]
    lane_stats = pd.DataFrame({label: coverage(train, validation, cols) for label, cols in [("Directed city lane", ["pickup", "delivery"]), ("Equipment", ["equipment"]), ("Directed lane + equipment", ["pickup", "delivery", "equipment"])]}).T
    sections += ["A lane is the ordered (pickup, delivery) pair; its reverse is a different lane. Counts below distinguish unique categories from affected rows.", table(lane_stats)]
    train_cities = set(cities(train).city)
    unseen = set(cities(validation).city) - train_cities
    unseen_rows = (~validation.pickup.isin(train_cities) | ~validation.delivery.isin(train_cities)).sum()
    sections.append(f"Validation has {len(unseen)} cities absent from train: {', '.join(sorted(unseen))}. They affect {unseen_rows:,} rows. Unseen lanes also include new combinations of known cities.")
    sections.append(table(pd.concat([train.equipment.value_counts().rename("Train rows"), validation.equipment.value_counts().rename("Validation rows")], axis=1)))
    return sections


def december_sections(train, raw_validation, december):
    sections = ["## December Chart Inputs"]
    sections.append(f"The chart file has {len(december)} rows and {len(december.columns)} columns. Its dates run from {december.date.min()} to {december.date.max()}, with {december.date.nunique()} distinct dates. predicted_rate has {december.predicted_rate.isna().sum()} blank placeholders. The missing training feature columns are {', '.join(c for c in raw_validation.columns if c not in december.columns)}.")
    fixed = december[["pickup", "delivery", "distance", "equipment", "weight"]].drop_duplicates()
    sections.append(table(fixed))
    city_map = cities(train).drop_duplicates()
    needed_cities = set(december.pickup) | set(december.delivery)
    lookup = city_map[city_map.city.isin(needed_cities)].set_index("city")
    sections.append(table(lookup))
    dec_lanes = pd.MultiIndex.from_frame(december[["pickup", "delivery"]].drop_duplicates())
    matched = pd.MultiIndex.from_frame(train[["pickup", "delivery"]]).isin(dec_lanes)
    matched_equipment = pd.MultiIndex.from_frame(train[["pickup", "delivery", "equipment"]]).isin(pd.MultiIndex.from_frame(december[["pickup", "delivery", "equipment"]].drop_duplicates()))
    sections.append(f"The {len(dec_lanes)} December directed lane(s) have {matched.sum():,} training rows, including {matched_equipment.sum():,} rows matching chart equipment. Train supplies {len(lookup)} coordinate mappings for {len(needed_cities)} required cities. This permits an internal lookup, subject to the coordinate-consistency limitation above.")
    sections.append("Neither market_index nor quote_signal is supplied for any of the 31 chart rows. Use a separate model trained and time-validated without those two features, using the six supplied inputs and train-derived coordinates. Do not invent future signals or copy validation signals from other loads. Validate this exact missing-signal setting before refitting. A chart may be flat if selected features do not support day-to-day changes; the input alone supplies no daily market observations.")
    return sections


def plan_sections(train, december):
    fit = train.date < "2025-09-01"
    tune = train.date.between("2025-09-01", "2025-09-30")
    final = train.date >= "2025-10-01"
    by_month = monthly_correlations(train)
    needed = set(december.pickup) | set(december.delivery)
    mappings = cities(train).drop_duplicates().city.isin(needed).sum()
    return ["## Recommended Plan",
        f"1. Fit {fit.sum():,} rows through August, tune on {tune.sum():,} September rows, then test once on {final.sum():,} October rows; also assess the combined two-month horizon.",
        "2. Use distance, equipment, positive/imputed weight, directed cities, coordinates and calendar features. Exclude load_id; the two files share zero IDs.",
        f"3. Compare a training median baseline and equipment RPM-times-distance baseline; distance has rate Pearson correlation {train.distance.corr(train.posted_rate):.4f}.",
        "4. Compare Ridge with encoded categories against HistGradientBoostingRegressor; select by time-holdout MAE and RMSE, including seen/unseen-lane slices.",
        f"5. Report the signal-free model as the headline. Treat quote_signal as a leak unless provenance proves it predates the quote; monthly RPM Spearman spans {by_month['Quote vs RPM Spearman'].min():.4f} to {by_month['Quote vs RPM Spearman'].max():.4f}. Keep signal ablations diagnostic, with market-only deployment conditional on provenance and time validation.",
        f"6. For all {len(december)} December dates, use the separately validated signal-free model and the {mappings} train coordinate mappings; refit selected models through October."]


def make_report(raw_train, raw_validation, december):
    train, validation = prepare(raw_train), prepare(raw_validation)
    sections = completeness_sections(raw_train, raw_validation, train, validation)
    sections += ["Reproduce from the worktree with the supplied Python environment:",
                 '```powershell\n& D:\\Assessments\\Spotter\\freight-rate\\.venv\\Scripts\\python.exe src/eda.py --data-dir D:\\Assessments\\Spotter\\freight-rate\\data\n```',
                 "The default --data-dir is data. The script writes this report and four aggregate figures under reports/."]
    sections += time_sections(train, validation)
    target, outliers = target_sections(train)
    sections += target
    sections += signal_sections(train, validation)
    numeric_columns = raw_train.select_dtypes('number').columns.intersection(raw_validation.select_dtypes('number').columns)
    sections += quality_sections(train, validation, outliers, numeric_columns)
    sections += coverage_sections(train, validation)
    sections += december_sections(train, raw_validation, december)
    sections += plan_sections(train, december)
    return "\n\n".join(sections) + "\n", train, validation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args()
    frames = [pd.read_csv(args.data_dir / name) for name in ["train_test.csv", "validation.csv", "december_chart_inputs.csv"]]
    report, train, validation = make_report(*frames)
    output = Path(__file__).resolve().parents[1] / "reports"
    save_charts(train, validation, output / "figures")
    (output / "eda_findings.md").write_text(report, encoding="utf-8")
    print(f"Train: {len(train):,} rows; Validation: {len(validation):,} rows; December: {len(frames[2]):,} rows")
    print("Wrote reports/eda_findings.md and four labelled PNG figures.")


if __name__ == "__main__":
    main()
