"""Read the quote_signal regime of each month from unlabeled columns only, then check it against the truth."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from eda import table

ROOT = Path(__file__).resolve().parents[1]


def regime_statistic(month):
    """Spearman of quote_signal with distance inside each equipment type, averaged. No posted_rate used."""
    values = [group.quote_signal.corr(group.distance, method="spearman") for _, group in month.groupby("equipment")]
    return float(np.mean(values))


def true_regime(month):
    """Label from posted_rate, used only to check the detector (and to fix thresholds on January to August)."""
    corr = (month.posted_rate / month.distance).corr(month.quote_signal, method="spearman")
    return "direct" if corr > .8 else "mirror" if corr < -.8 else "unresolved"


def detect(statistic, threshold):
    """In a direct month quote follows rate per mile, which falls with distance, so the statistic is negative."""
    if abs(statistic) < threshold:
        return "unresolved"
    return "direct" if statistic < 0 else "mirror"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args()
    train = pd.read_csv(args.data_dir / "train_test.csv", parse_dates=["date"])
    validation = pd.read_csv(args.data_dir / "validation.csv", parse_dates=["date"])

    rows = []
    for month, group in train.groupby(train.date.dt.month):
        rows.append({"month": f"2025-{month:02d}", "rows": len(group),
                     "statistic": regime_statistic(group), "true_regime": true_regime(group)})
    development = pd.DataFrame(rows)
    early = development.iloc[:8]
    resolved = early[early.true_regime != "unresolved"].statistic.abs()
    threshold = float(resolved.min() / 2)  # fixed on January to August only
    development["detected"] = development.statistic.map(lambda s: detect(s, threshold))
    development["correct"] = development.detected == development.true_regime
    held_out = development.iloc[8:]

    future = []
    for month, group in validation.groupby(validation.date.dt.month):
        weekly = group.groupby(group.date.dt.isocalendar().week).apply(
            lambda week: week.quote_signal.corr(week.distance, method="spearman"), include_groups=False)
        future.append({"month": f"2025-{month:02d}", "rows": len(group), "statistic": regime_statistic(group),
                       "largest_weekly_abs_correlation": float(weekly.abs().max()),
                       "quote_mean": group.quote_signal.mean(), "quote_std": group.quote_signal.std()})
    future = pd.DataFrame(future)
    future["detected"] = future.statistic.map(lambda s: detect(s, threshold))
    august = train[train.date.dt.month == 8].quote_signal
    august_row = {"quote_mean": august.mean(), "quote_std": august.std(), "quote_min": august.min(), "quote_max": august.max()}

    summary = {"threshold": threshold, "development_accuracy": float(development.correct.mean()),
               "held_out_accuracy": float(held_out.correct.mean()),
               "validation_detected": future.set_index("month").detected.to_dict()}
    (ROOT / "outputs").mkdir(exist_ok=True)
    (ROOT / "outputs/regime_detection.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    all_unresolved = future.detected.eq("unresolved").all()
    sections = [
        "# Quote Signal Regime Detection",
        "The monthly regime of quote_signal is read from columns that also exist in validation, without posted_rate. "
        "In a direct month quote_signal follows rate per mile, which falls as distance rises, so quote_signal and distance "
        "are negatively correlated. In a mirrored month the relation flips and the correlation is positive. In an unresolved month "
        "(August) quote_signal is noise and the correlation is near zero. The statistic is the Spearman correlation of "
        "quote_signal with distance inside each equipment type, averaged over the three types.",
        f"The threshold is half of the smallest absolute statistic among the resolved months from January to August, which is {threshold:.3f}. "
        "It was fixed on January to August only and applied unchanged to September and October.",
        "## Development Months (true regime shown only to check the detector)",
        table(development.drop(columns=["correct"]).assign(correct=development.correct.map({True: "yes", False: "no"}))),
        f"Detection accuracy: {development.correct.mean() * 100:.0f} percent over January to October, and {held_out.correct.mean() * 100:.0f} percent on the two held-out months (September and October).",
        "## Validation Months (no labels)",
        table(future),
        f"For comparison, August (the unresolved month) has quote_signal mean {august_row['quote_mean']:.3f}, standard deviation "
        f"{august_row['quote_std']:.3f}, minimum {august_row['quote_min']:.3f} and maximum {august_row['quote_max']:.3f}.",
        ("## Conclusion",
         "Both validation months are unresolved: the statistic is near zero and no week reaches the size of a resolved month, and "
         "the quote_signal distribution is August-like. In other words quote_signal is noise in November and December. "
         "A regime-aware model has nothing to use, so no alternative prediction file is written and quote_signal stays excluded "
         "from the final model. This confirms the earlier decision with a direct check instead of a classifier."
         if all_unresolved else
         "At least one validation month is resolved. See the table above before deciding whether a regime-aware model is allowed; "
         "quote_signal is close to the target in direct months, so its provenance must be confirmed first."),
    ]
    flat = []
    for item in sections:
        flat.extend(item if isinstance(item, tuple) else [item])
    (ROOT / "reports/quote_regime_detection.md").write_text("\n\n".join(flat) + "\n", encoding="utf-8")
    print(development.to_string(index=False))
    print(future.to_string(index=False))
    print(summary)


if __name__ == "__main__":
    main()
