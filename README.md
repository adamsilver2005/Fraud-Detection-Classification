# Fraud Detection: Multi-Class Urgency Classification

A machine learning project that predicts the **investigation urgency level** of financial transactions, ranging from no action required to immediate intervention.

Originally built as a team submission for **HackML 2026** on Kaggle. This repository is my personal continuation of that work, with refactored code, improved structure, and ongoing experimentation.


## Problem Statement

Financial institutions process millions of transactions daily, yet only a small fraction are fraudulent. Rather than asking *"Is this fraud?"*, fraud teams must decide **how urgently** a transaction needs investigation — given limited analyst resources.

**Type:** Supervised Multi-Class Classification  
**Target Variable:** `urgency_level` (0–3)

| Label | Description | Business Context |
|-------|-------------|-----------------|
| 0 | No Action | Transaction appears legitimate |
| 1 | Monitor | Low-risk suspicious activity |
| 2 | Review | Likely fraud requiring analyst review |
| 3 | Immediate Action | High-risk fraud requiring urgent response |

The dataset is intentionally **imbalanced**, reflecting real-world fraud distributions. Models are evaluated on **Macro F1-score**, which treats all urgency levels equally and prevents ignoring rare but critical fraud cases.

---

## Project Structure

```
fraud-detection-classification/
│
├── data/                         # Train and test
|   outputs/                       # Outputs and catboost info
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

---

## Approach

Feature Engineering: (`jupyter_notebook_files/data_preprocessing.ipynb`)
- Balance discrepancy checks per transaction type (tolerant float comparison)
- Delta features: `deltaOrg`, `deltaDest`
- Large transaction flag using 99.9th percentile threshold
- Merchant destination flag

Models:
- CatBoost (`jupyter_notebook_files/model_train.ipynb`) — native categorical support, early stopping
- XGBoost (`python_files/xgtry.py`) — one-hot encoded, time-based validation split

Class Imbalance Handling:
- Inverse-frequency class/sample weights
- Macro F1 as the primary evaluation metric

---

## Results

| Model | Validation Macro F1 |
|-------|-------------------|
| CatBoost | TBD |
| XGBoost | TBD |

---

## Getting Started

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

---

## Attribution

This project originated as a team submission for **HackML 2026** on Kaggle, built collaboratively with [@Peyton289](https://github.com/Peyton289), [@TrentB159](https://github.com/TrentB159), and [ArwinSepahram](https://github.com/ArwinSepahram).

This repository is my personal continuation of that work with independent improvements and refactoring.

**Competition:** [FRAUD | HackML 2026](https://kaggle.com/competitions/fraud-hack-ml-2026)  

