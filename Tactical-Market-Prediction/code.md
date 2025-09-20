The Kaggle competition **"Hull Tactical - Market Prediction"** challenges participants to **predict the daily returns of the S\&P 500 index** using a provided, tailored set of market data features.

Here's a breakdown of the competition's data, phases, and files in Markdown format:

## Dataset Description

This competition challenges you to predict the daily returns of the **S\&P 500 index** using a tailored set of market data features.

---

## Competition Phases and Data Updates

The competition will proceed in two phases:

1.  A **model training phase** with a test set of six months of historical data. Because these prices are publicly available, ***leaderboard scores during this phase are not meaningful***.
2.  A **forecasting phase** with a test set to be collected *after* submissions close. The scored portion of this test set is expected to be about the same size as the scored portion of the first phase.

During the forecasting phase, the evaluation API will serve test data from the beginning of the public set to the end of the private set. The first `date_id` served by the API will remain constant throughout the competition.

---

## Files

### `train.csv`

Historic market data, stretching back decades, with extensive missing values expected early on.

| Column Name | Description |
| :--- | :--- |
| **`date_id`** | An identifier for a single trading day. |
| **`M*`** | Market Dynamics/Technical features. |
| **`E*`** | Macro Economic features. |
| **`I*`** | Interest Rate features. |
| **`P*`** | Price/Valuation features. |
| **`V*`** | Volatility features. |
| **`S*`** | Sentiment features. |
| **`MOM*`** | Momentum features. |
| **`D*`** | Dummy/Binary features. |
| **`forward_returns`** | The returns from buying the S\&P 500 and selling it a day later. *(Train set only)* |
| **`risk_free_rate`** | The federal funds rate. *(Train set only)* |
| **`market_forward_excess_returns`** | Forward returns relative to expectations. Computed by subtracting the rolling five-year mean forward returns and winsorizing the result using a median absolute deviation (MAD) with a criterion of 4. *(Train set only)* |

### `test.csv`

A mock test set representing the structure of the unseen test set. The test set used for the **public leaderboard is a copy of the last 180 `date_id`s in the train set**, meaning ***the public leaderboard scores are not meaningful***.

| Column Name | Description |
| :--- | :--- |
| **`date_id`** | An identifier for a single trading day. |
| **`[feature_name]`** | The feature columns are the same as in `train.csv`. |
| **`is_scored`** | Whether this row is included in the evaluation metric calculation. During the model training phase this will be **true for the first 180 rows only**. |
| **`lagged_forward_returns`** | The returns from buying the S\&P 500 and selling it a day later, provided with a lag of one day. |
| **`lagged_risk_free_rate`** | The federal funds rate, provided with a lag of one day. |
| **`lagged_market_forward_excess_returns`** | Forward returns relative to expectations. Computed by subtracting the rolling five-year mean forward returns and winsorizing the result using a median absolute deviation (MAD) with a criterion of 4, provided with a lag of one day. |

### `kaggle_evaluation/`

Files used by the **evaluation API**. See [the demo submission](https://www.kaggle.com/code/sohier/hull-tactical-market-prediction-demo-submission) for an illustration of how to use the API.

---

## Data Summary

* **Total Files:** 13 files
* **Total Size:** 12.39 MB
* **Types:** py, csv, proto
* **License:** Subject to Competition Rules
* **Columns (Total):** 197 columns
* **Download Command (Kaggle API):** `kaggle competitions download -c hull-tactical-market-prediction`