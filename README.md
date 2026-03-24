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



## Project Structure

```
fraud-detection-classification/
│
├── data/                         # Train and test CSVs
├── outputs/                      # Generated files and CatBoost logs
├── jupyter_notebook_files/
│   ├── data_preprocessing.ipynb  # Feature engineering pipeline
│   └── model_train.ipynb         # CatBoost training and evaluation
├── python_files/
│   ├── data_fraud.py             # EDA and exploratory analysis
│   └── xgtry.py                  # XGBoost model experiment
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Exploratory Data Analysis

Initial analysis performed in `python_files/data_fraud.py`:

- Investigated class distribution across urgency levels — the dataset is heavily skewed toward class 0 (no action), reflecting real-world fraud rarity
- Identified accounts appearing across multiple time steps to detect repeat offenders
- Analyzed transaction amounts for flagged transactions (urgency 1–3) — fraudulent transactions tend to have significantly higher mean amounts than legitimate ones
- Examined transaction type breakdowns across urgency levels

Key feature engineering decisions made from EDA (`jupyter_notebook_files/data_preprocessing.ipynb`):
- Balance discrepancy flags per transaction type using tolerant float comparison
- Delta features (`deltaOrg`, `deltaDest`) to capture net flow per transaction
- Large transaction flag based on the 99.9th percentile amount threshold
- Merchant destination flag derived from `nameDest` prefix

---

## Predictive Modeling

Two gradient boosting models were trained and evaluated using a **time-based train/validation split** (last 20% of time steps held out for validation).

Class Imbalance Handling:
- Inverse-frequency sample weights applied during training
- Macro F1-score used as the primary evaluation metric to equally weight all urgency levels

CatBoost (`jupyter_notebook_files/model_train.ipynb`)
- Native handling of categorical features (no encoding needed)
- Early stopping with 200-iteration patience
- Tuned depth, L2 regularization, and learning rate

XGBoost (`python_files/xgtry.py`)
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
Notebooks can be run in order: `data_preprocessing.ipynb` → `model_train.ipynb`.

---

## Attribution

This project originated as a team submission for **HackML 2026** on Kaggle, built collaboratively with [Peyton289](https://github.com/Peyton289), [TrentB159](https://github.com/TrentB159), and [ArwinSepahram](https://github.com/ArwinSepahram).

This repository is my personal continuation of that work with independent improvements and refactoring.

**Competition:** [FRAUD | HackML 2026](https://kaggle.com/competitions/fraud-hack-ml-2026)




