# Freight Rate Prediction

Solution to the Spotter machine learning assessment: predict the rate of a freight load from its lane, distance, equipment,
weight and date, and produce the December chart.

## Result

- Model: gradient boosted trees (scikit-learn `HistGradientBoostingRegressor`, absolute error loss) predicting rate per mile,
  multiplied by distance. No market signals.
- October holdout (choices frozen on September first): mean absolute error about 109, typical error about 5 percent, R squared 0.82.
- The December chart uses a separate model that only uses the inputs the chart file contains.
- Details and tables: `reports/model_comparison.md`. Data findings: `reports/eda_findings.md`.
  How `quote_signal` was handled: `reports/quote_signal_regimes.md` and `reports/quote_regime_detection.md`.

## Setup

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt     # on Linux or macOS: .venv/bin/python
```

Put the data from the assessment in a `data` folder (it is not in this repository because it is private):

```
data/train_test.csv
data/validation.csv
data/validation_predictions_template.csv
data/december_chart_inputs.csv
```

## Run

Run these from the repository root. Every script takes `--data-dir` (default `data`).

```bash
python src/eda.py                 # data checks and charts -> reports/eda_findings.md
python src/quote_regimes.py       # quote_signal study using labels (development months only)
python src/regime_detect.py       # label-free regime detection -> reports/quote_regime_detection.md
python src/train.py               # tune on September, evaluate October, write predictions and the report
python score.py --predictions validation_predictions.csv --december-predictions outputs/december_chart_inputs_predicted.csv
```

The scorer checks both files and writes `scorer_results/candidate_december.png`. `train.py` takes a while because it fits a
few dozen candidate models. Unit tests: `python -m unittest discover -s tests`.

## Files

| Path | What it does |
| --- | --- |
| `src/features.py` | Feature builder fitted inside each training window (weight fill, coordinates, calendar, distance ratio) |
| `src/train.py` | Time-based tuning, frozen choices, October holdout, final predictions, December model |
| `src/regime_detect.py` | Label-free detection of the quote_signal regime of each month |
| `src/quote_regimes.py` | Development-month study of quote_signal against the rate per mile |
| `src/eda.py` | Data checks |
| `validation_predictions.csv` | The 12,000 predictions (`load_id,predicted_rate`) |
| `outputs/december_chart_inputs_predicted.csv` | The 31 December predictions |
| `outputs/*.csv`, `outputs/model_choices.json` | Candidate scores, holdout metrics and the frozen choices |

## Notes on the approach

- The validation file is later than the training file, so validation is time-based: fit January to August, tune on September,
  evaluate October once, then refit on everything.
- The supplied distance is used as is. The loads that look suspicious against the city coordinates are 70 mile loads, the smallest
  distance in the data, and they sold for about 200 to 230, so the coordinates are the approximate part.
- `quote_signal` is not used: it follows the target in some months, mirrors it in others and is noise in August, and the label-free
  detector finds November and December in the validation file to be noise as well.
