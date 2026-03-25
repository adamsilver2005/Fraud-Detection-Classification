# Fraud Detection: Multi-Class Urgency Classification

A machine learning project that predicts the **investigation urgency level** of financial transactions, ranging from no action required to immediate intervention.

Originally built as a team submission for **HackML 2026** on Kaggle. This repository is my personal continuation of that work, with refactored code, improved structure, and ongoing experimentation.


## Table of Contents
- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Predictive Modeling](#predictive-modeling)
- [Final Conclusion](#final-conclusion)
- [How to Run the Project](#how-to-run-the-project)


## Problem Statement

Financial institutions process millions of transactions daily, yet only a small fraction are fraudulent. Rather than asking *"Is this fraud?"*, fraud teams must decide **how urgently** a transaction needs investigation, given limited analyst resources.

**Type:** Supervised Multi-Class Classification  
**Target Variable:** `urgency_level` (0–3)

| Label | Description | Business Context |
|-------|-------------|-----------------|
| 0 | No Action | Transaction appears legitimate |
| 1 | Monitor | Low-risk suspicious activity |
| 2 | Review | Likely fraud requiring analyst review |
| 3 | Immediate Action | High-risk fraud requiring urgent response |

The dataset is intentionally **imbalanced**, reflecting real-world fraud distributions. Models are evaluated on **Macro F1-score**, which treats all urgency levels equally and prevents ignoring rare but critical fraud cases.


## Dataset

Each row represents a single anonymized transaction from a simulated payment system (PaySim).

**Raw features (`data/train.csv`):**

| Feature | Description |
|---------|-------------|
| `step` | Time step (1 step = 1 hour) |
| `type` | Transaction type: `CASH_IN`, `CASH_OUT`, `DEBIT`, `PAYMENT`, `TRANSFER` |
| `amount` | Transaction amount in local currency |
| `oldbalanceOrg` | Origin account balance before transaction |
| `newbalanceOrig` | Origin account balance after transaction |
| `oldbalanceDest` | Destination account balance before transaction |
| `newbalanceDest` | Destination account balance after transaction |
| `nameOrig` | Anonymized origin account ID |
| `nameDest` | Anonymized destination account ID |
| `urgency_level` | **Target**: investigation urgency (0–3) |

**Engineered features (`outputs/train_with_new_features.csv`):**

After running `data_preprocessing.ipynb`, 6 additional columns are added to the raw data:

| Feature | Description |
|---------|-------------|
| `check_balanceOrg_tol` | Flag — did the origin balance not update correctly for this transaction type? |
| `check_balanceDest_tol` | Flag — did the destination balance not update correctly? |
| `check_Amount_Size` | Flag — is this transaction in the top 0.1% by amount? |
| `deltaOrg` | Net change in origin account balance |
| `deltaDest` | Net change in destination account balance |
| `dest_is_merchant` | Flag — is the destination a merchant account? |

The enriched CSVs (`train_with_new_features.csv`, `test_with_new_features.csv`) are saved to `outputs/` and used as input for model training.


## Project Structure

```

fraud-detection-classification/
│
├── data/                         # Train and test CSVs
├── outputs/                      # Generated files and CatBoost logs
├── jupyter_notebook_files/
│   ├── eda.ipynb                 # Exploratory data analysis and visualizations
│   ├── data_preprocessing.ipynb  # Feature engineering pipeline
│   └── model_train.ipynb         # CatBoost training and evaluation
├── python_files/
│   ├── data_fraud.py             # EDA and exploratory analysis
│   └── xgtry.py                  # XGBoost model experiment
├── requirements.txt
├── .gitignore
└── README.md
```

## Exploratory Data Analysis

Full EDA with code and visualizations is in `jupyter_notebook_files/eda.ipynb`.


### Class Distribution

![Class Distribution](outputs/plot_class_distribution.png)

The dataset is extremely imbalanced. Out of 6,244,474 total transactions, 99.89% are class 0 (No Action). The three fraud classes each account for only 0.03–0.04% of the data:

| Urgency Level | Count | Share |
|---|---|---|
| 0 — No Action | 6,237,903 | 99.89% |
| 1 — Monitor | 2,176 | 0.03% |
| 2 — Review | 2,151 | 0.03% |
| 3 — Immediate Action | 2,244 | 0.04% |

This extreme imbalance means a naive model that always predicts class 0 would achieve 99.89% accuracy while being completely useless for fraud detection. This is why Macro F1-score is the evaluation metric of choice, as it computes F1 independently for each class and averages them, treating all urgency levels equally regardless of how rare they are. It also means that class weights or sample weights are needed during training to prevent the model from simply ignoring the minority classes.


### Transaction Type Breakdowns

![Transaction Types](outputs/plot_transaction_types.png)

`CASH_OUT` is the most common transaction type (2,200,304), followed closely by `PAYMENT` (2,110,276) and `CASH_IN` (1,372,041). `TRANSFER` (521,363) and `DEBIT` (40,490) are far less frequent.

The stacked percentage chart shows that all transaction types are nearly 100% class 0, reflecting the overall dataset imbalance. However, the small amount of fraud that does exist is spread across all types rather than concentrated in one. This means that `type` alone cannot identify fraud, but it remains an important feature when combined with other signals. In particular, `TRANSFER` and `CASH_OUT` transactions involve direct money movement between accounts, making them inherently higher risk than `CASH_IN` or `PAYMENT`.


### Amount Distributions by Urgency Level

![Amount Distributions](outputs/plot_amount_distributions.png)
![Amount Box Plot](outputs/plot_amount_boxplot.png)

Transaction amounts increase dramatically with urgency level:

| Urgency Level | Mean Amount | Median Amount |
|---|---|---|
| 0 — No Action | 178,608 | 74,835 |
| 1 — Monitor | 80,406 | 73,524 |
| 2 — Review | 478,715 | 436,035 |
| 3 — Immediate Action | 3,626,451 | 2,392,483 |

Class 3 (Immediate Action) transactions have a mean amount of 3.6 million, which is roughly 20x higher than class 0. Notably, class 1 (Monitor) has a lower mean than class 0, suggesting that low-level suspicious activity does not necessarily involve large amounts. Classes 2 and 3 show a sharp jump, indicating that high-value transactions are a strong signal for serious fraud.

All distributions are right-skewed, so a small number of very large transactions pull the mean above the median. The gap between mean and median is largest for class 3 (3.6M vs 2.4M), indicating that extreme outlier amounts are especially common among the most urgent fraud cases. This justifies engineering a large transaction flag at the 99.9th percentile threshold as a feature.


### Correlation Heatmap

![Correlation Heatmap](outputs/plot_correlation_heatmap.png)

Looking at the bottom row (`urgency_level`), the features most correlated with the target are:

| Feature | Correlation with `urgency_level` |
|---|---|
| `deltaOrg` | 0.44 |
| `check_Amount_Size` | 0.05 |
| `dest_is_merchant` | -0.02 |
| `step` | 0.01 |

`deltaOrg` (net outflow from the origin account) is by far the strongest predictor at 0.44, confirming that large money movements out of an account are the clearest fraud signal in this dataset. `check_Amount_Size` and `dest_is_merchant` are weaker but still directionally useful.

Two notable multicollinearity findings: `oldbalanceDest` and `newbalanceDest` are correlated at 0.98, meaning they carry almost identical information and one is largely redundant. Similarly, `deltaOrg` and `deltaDest` capture the net balance change more cleanly than the raw balance columns, which is why they were engineered as features. `step` (time) has near-zero correlation with urgency, confirming that fraud occurs consistently across all time periods rather than spiking at particular hours.


### Balance Discrepancy Analysis

![Balance Discrepancy](outputs/plot_balance_discrepancy.png)

For each transaction type, there is an expected relationship between the old balance, the transaction amount, and the new balance, for example, a `CASH_OUT` should decrease the origin balance by exactly `amount`. A discrepancy flags where this relationship does not hold.

For the origin balance, class 0 has a discrepancy rate of ~57%, while classes 1–3 all drop to near 0%. This is a somewhat counterintuitive finding, it suggests that legitimate transactions are more likely to have minor balance irregularities (likely floating point rounding), while flagged transactions tend to have clean, exact balance updates. This may indicate that fraudulent transactions are carefully constructed to appear internally consistent.

For the destination balance, discrepancy rates are approximately 50/50 across classes 1–3, making it a weaker but still useful signal. Together, these flags add information that raw balance columns alone do not capture.


### Feature Engineering Pipeline

The raw data is enriched with 6 new features in `data_preprocessing.ipynb` and saved as `train_with_new_features.csv` and `test_with_new_features.csv` in `outputs/`. These enriched files are used as input for model training.

| Feature | Description | Motivation from EDA |
|---|---|---|
| `check_balanceOrg_tol` | Flag: origin balance did not update as expected | Discrepancy rate differs significantly across urgency levels |
| `check_balanceDest_tol` | Flag: destination balance did not update as expected | Adds signal for TRANSFER and DEBIT transactions |
| `deltaOrg` | Net change in origin account balance | Strongest single predictor of urgency (r = 0.44) |
| `deltaDest` | Net change in destination account balance | Less redundant than raw balance columns |
| `check_Amount_Size` | Flag: transaction is in the top 0.1% by amount | Class 3 mean amount is 20x higher than class 0 |
| `dest_is_merchant` | Flag: destination is a merchant account | Merchant destinations are associated with lower urgency |


## Predictive Modeling

Two gradient boosting models were trained and evaluated using a **time-based train/validation split** (last 20% of time steps held out for validation).

**Class Imbalance Handling:**
- Inverse-frequency sample weights applied during training
- Macro F1-score used as the primary evaluation metric to equally weight all urgency levels

**CatBoost** (`jupyter_notebook_files/model_train.ipynb`)
- Native handling of categorical features (no encoding needed)
- Early stopping with 200-iteration patience
- Tuned depth, L2 regularization, and learning rate

**XGBoost** (`python_files/xgtry.py`)
- One-hot encoded categorical features
- Per-row sample weights for imbalance handling
- Tuned max depth, subsampling, and regularization

| Model | Validation Macro F1 |
|-------|-------------------|
| CatBoost | TBD |
| XGBoost | TBD |

---

## Final Conclusion

*To be updated after final model experiments are complete.*

The project demonstrates that gradient boosting models with careful feature engineering and class imbalance handling are well-suited for fraud urgency classification. Balance discrepancy flags and delta features proved to be particularly informative signals.

---

## How to Run the Project

```bash
# Clone the repo
git clone https://github.com/adamsilver2005/Fraud-Detection-Classification.git
cd Fraud-Detection-Classification

# Install dependencies
pip install -r requirements.txt

# Run EDA
python python_files/data_fraud.py

# Run XGBoost model
python python_files/xgtry.py
```

Place `train.csv` and `test.csv` inside the `data/` folder before running.

Notebooks should be run in order:
1. `data_preprocessing.ipynb` — engineers new features and saves enriched CSVs to `outputs/`
2. `model_train.ipynb` — trains CatBoost on the enriched data and outputs a submission file

---

## Attribution

This project originated as a team submission for **HackML 2026** on Kaggle, built collaboratively with [Peyton289](https://github.com/Peyton289), [TrentB159](https://github.com/TrentB159), and [ArwinSepahram](https://github.com/ArwinSepahram).

This repository is my personal continuation of that work with independent improvements and refactoring.

**Competition:** [FRAUD | HackML 2026](https://kaggle.com/competitions/fraud-hack-ml-2026)