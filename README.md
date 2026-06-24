# 🧬 DataSci Studio Pro
An end-to-end, modular data-science pipeline: load → validate → clean → explore → model → export.
Available both as a command-line tool for reproducible, automated runs and as an interactive Streamlit dashboard.

Built to demonstrate the full data‑product lifecycle a data scientist is expected to own — ingestion, data quality, exploratory data analysis, modeling, evaluation, visualization, and delivery — not just a single notebook.

Unlike a one‑off analysis script, this project packages a repeatable pipeline that a non‑technical user can run from a dashboard (upload a file → receive cleaned data, charts, KPIs, a trained model, and a PDF report in minutes) and that an engineer can schedule from the command line (for example: python app.py --input sales.csv --target churn --report) as part of a daily reporting job — cutting manual reporting and ad‑hoc analysis time from hours to minutes.



✨ Features

| Area | What it does |
|---|---|
| **Data loading** | CSV, Excel (`.xlsx`/`.xls`), JSON, Parquet, and SQL (via SQLAlchemy) |
| **Validation** | Null counts/%, duplicate rows, dtypes, constant columns, high-cardinality columns, IQR outlier counts |
| **Cleaning** | Column-name standardization, high-null column dropping, duplicate removal, type inference (numeric/datetime), null imputation (median/mean/zero/ffill/drop), one-hot encoding, scaling (z-score/min-max/robust), IQR outlier capping |
| **EDA** | Summary statistics (incl. skew/kurtosis), correlation matrix & top correlated pairs, distribution grids, correlation heatmap, scatter (with trend line), box plots, bar & line charts |
| **Modeling** | Auto-detects classification vs. regression; trains Random Forest, Gradient Boosting, Logistic Regression, KNN, SVM (classification) or Random Forest, Gradient Boosting, Linear/Ridge/Lasso (regression); cross-validation; one-click "compare all models" leaderboard |
| **Evaluation** | Accuracy, precision, recall, F1 (classification); R², RMSE, MAE (regression); confusion matrix and feature-importance charts |
| **Export** | Cleaned data → CSV / Excel (with a stats sheet) / JSON; charts → PNG; trained model → `.joblib`; full multi-section **PDF report** |
| **Dashboard** | KPI cards, sidebar filters, tabbed workflow (Data → Clean → EDA → Model → Export), consistent dark theme |
| **Ops** | Rotating file + console logging, CLI with `argparse`, structured JSON reports for every run |

---

## 🗂️ Project structure

```
datasci-studio-pro/
├── app.py              # CLI entry point — runs the full pipeline end-to-end
├── dashboard.py         # Streamlit UI — interactive version of the same pipeline
├── data_loader.py        # CSV / Excel / JSON / Parquet / SQL ingestion
├── cleaning.py            # Validation + cleaning/preprocessing functions
├── eda.py                  # Summary stats, correlations, chart generation
├── model.py                 # Train/evaluate/save classification & regression models
├── exporter.py               # CSV/Excel/PNG/model/PDF export helpers
├── logger_config.py           # Rotating file + console logging setup
├── requirements.txt
├── data/
│   └── retail_sales.csv       # Sample dataset (2,000 synthetic orders)
├── docs/screenshots/            # Example chart output (see below)
├── exports/                       # Pipeline run outputs (created at runtime)
├── models/                          # Saved .joblib models (created at runtime)
└── logs/                              # Rotating log files (created at runtime)
```

Each module is **pure and composable** — every cleaning/EDA/model function
takes a DataFrame in and returns a DataFrame (or figure/dict) out, with no
hidden global state. That means:

- `app.py` and `dashboard.py` are thin orchestration layers over the same
  five modules — fix a bug once, both interfaces benefit.
- Any function can be imported and unit-tested or reused in a notebook.

---

## 🚀 Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the full pipeline on the sample dataset (no modeling)
python app.py --input data/retail_sales.csv

# 3. Or launch the interactive dashboard
streamlit run dashboard.py
```

---

## 🖥️ CLI usage (`app.py`)

```bash
# Clean + profile a dataset, export cleaned CSV/Excel + EDA charts
python app.py --input data/retail_sales.csv

# Train a classifier to predict 'returned', save the model, build a PDF report
python app.py --input data/retail_sales.csv --target returned --report

# Regression target, compare every available model, custom output folder
python app.py --input data/retail_sales.csv --target profit \
               --task regression --compare-models \
               --output-dir exports/profit_run

# Pull data from a database instead of a file
python app.py --sql-query "SELECT * FROM sales" \
               --sql-conn "postgresql://user:pass@host/db" --target profit
```

Every run writes a self-contained folder (default `exports/`) containing:

```
exports/
├── validation_report.json     # data-quality report
├── cleaning_report.json       # steps applied + before/after shape
├── summary_statistics.csv     # describe() + skew/kurtosis/nulls
├── top_correlations.csv
├── cleaned_data.csv / .xlsx
├── charts/
│   ├── distributions.png
│   ├── correlation_heatmap.png
│   └── box_plot.png
├── model/                      # only if --target is set
│   ├── metrics.json
│   ├── feature_importance.png
│   ├── confusion_matrix.png    # classification only
│   ├── model_comparison.csv    # only with --compare-models
│   └── random_forest.joblib
└── report.pdf                  # only with --report
```

Run `python app.py --help` for the full list of options (null-handling
strategy, outlier capping, test split size, model choice, etc.).

---

## 📊 Dashboard (`dashboard.py`)

```bash
streamlit run dashboard.py
```

The dashboard mirrors the CLI pipeline as a guided workflow:

1. **Sidebar** — upload a dataset, apply categorical/numeric filters, and
   set global cleaning/test-split settings.
2. **KPI bar** — total rows, feature count, missing-value %, duplicate count.
3. **Data tab** — preview, column info, and a one-click validation report.
4. **Clean tab** — run the auto-clean pipeline and compare before/after.
5. **EDA tab** — summary stats, top correlations, and a chart builder
   (distributions, heatmap, scatter, box, bar, line).
6. **Model tab** — pick a target + model, train or compare a full
   leaderboard, view metrics, feature importance, and confusion matrix.
7. **Export tab** — download CSV/Excel/JSON, chart PNGs, the trained model,
   and a full PDF report.

---

## 🖼️ Example output

Generated from the included `data/retail_sales.csv` sample dataset.

**Feature distributions**

![Distributions](docs/screenshots/distributions.png)

**Correlation heatmap**

![Correlation heatmap](docs/screenshots/correlation_heatmap.png)

**Feature importance (Random Forest classifier predicting `returned`)**

![Feature importance](docs/screenshots/feature_importance.png)

**Confusion matrix**

![Confusion matrix](docs/screenshots/confusion_matrix.png)

---

## 📁 Sample dataset

`data/retail_sales.csv` is a synthetic 2,000-row retail orders dataset
(region, customer segment, product category, pricing, profit, ratings,
return flag) that intentionally includes missing values, duplicate rows,
and outliers — so the cleaning and validation steps have something to do.
Suggested targets:

- `returned` (binary) → classification
- `profit` (continuous) → regression
- `product_category` (3 classes) → multi-class classification

Swap in your own CSV/Excel/JSON/Parquet file with `--input <path>` or the
dashboard's file uploader — no code changes needed.

---

## 🧰 Tech stack

`pandas` · `numpy` · `scikit-learn` · `matplotlib` / `seaborn` ·
`streamlit` · `joblib` · `reportlab` · `SQLAlchemy` · `argparse` / `logging`

---

## 🔧 Notes / design decisions

- **ID columns are auto-excluded** from model features (e.g. `order_id`) to
  avoid leakage and noisy "importance" scores.
- **Outlier capping skips binary/flag columns** (e.g. a 0/1 `returned`
  column) — capping a column whose IQR is zero would otherwise collapse it
  to a single value.
- **The modeling target is never one-hot-encoded away** — categorical
  targets (e.g. `product_category`) remain usable after encoding.
- All logs are written to `logs/` with daily rotation (5 MB × 3 backups),
  so production runs are auditable.
