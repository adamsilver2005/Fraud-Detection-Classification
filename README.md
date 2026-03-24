Fraud Detection: Multi-Class Urgency Classification
A machine learning project that predicts the investigation urgency level of financial transactions, ranging from no action required to immediate intervention.
Originally built as a team submission for HackML 2026 on Kaggle. This repository is my personal continuation of that work, with refactored code, improved structure, and ongoing experimentation.

Table of Contents

Problem Statement
Dataset
Project Structure
Exploratory Data Analysis
Predictive Modeling
Final Conclusion
How to Run the Project


Problem Statement
Financial institutions process millions of transactions daily, yet only a small fraction are fraudulent. Rather than asking "Is this fraud?", fraud teams must decide how urgently a transaction needs investigation, given limited analyst resources.
Type: Supervised Multi-Class Classification
Target Variable: urgency_level (0–3)
LabelDescriptionBusiness Context0No ActionTransaction appears legitimate1MonitorLow-risk suspicious activity2ReviewLikely fraud requiring analyst review3Immediate ActionHigh-risk fraud requiring urgent response
The dataset is intentionally imbalanced, reflecting real-world fraud distributions. Models are evaluated on Macro F1-score, which treats all urgency levels equally and prevents ignoring rare but critical fraud cases.

Dataset
Each row represents a single anonymized transaction from a simulated payment system (PaySim).
Raw features (data/train.csv):
FeatureDescriptionstepTime step (1 step = 1 hour)typeTransaction type: CASH_IN, CASH_OUT, DEBIT, PAYMENT, TRANSFERamountTransaction amount in local currencyoldbalanceOrgOrigin account balance before transactionnewbalanceOrigOrigin account balance after transactionoldbalanceDestDestination account balance before transactionnewbalanceDestDestination account balance after transactionnameOrigAnonymized origin account IDnameDestAnonymized destination account IDurgency_levelTarget: investigation urgency (0–3)
Engineered features (outputs/train_with_new_features.csv):
After running data_preprocessing.ipynb, 6 additional columns are added to the raw data:
FeatureDescriptioncheck_balanceOrg_tolFlag — did the origin balance not update correctly for this transaction type?check_balanceDest_tolFlag — did the destination balance not update correctly?check_Amount_SizeFlag — is this transaction in the top 0.1% by amount?deltaOrgNet change in origin account balancedeltaDestNet change in destination account balancedest_is_merchantFlag — is the destination a merchant account?
The enriched CSVs (train_with_new_features.csv, test_with_new_features.csv) are saved to outputs/ and used as input for model training.

Project Structure
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

Exploratory Data Analysis
Initial analysis performed in python_files/data_fraud.py:

Investigated class distribution across urgency levels — the dataset is heavily skewed toward class 0 (no action), reflecting real-world fraud rarity
Identified accounts appearing across multiple time steps to detect repeat offenders
Analyzed transaction amounts for flagged transactions (urgency 1–3) — fraudulent transactions tend to have significantly higher mean amounts than legitimate ones
Examined transaction type breakdowns across urgency levels

Feature engineering pipeline (jupyter_notebook_files/data_preprocessing.ipynb):
The raw data is enriched with 6 new features and saved as train_with_new_features.csv and test_with_new_features.csv in the outputs/ folder. These enriched files are then used as input for model training.

Balance discrepancy flags — checks whether origin/destination balances updated correctly for each transaction type (e.g. a CASH_OUT should decrease the origin balance by amount). Uses tolerant float comparison to avoid false positives from rounding errors
Delta features (deltaOrg, deltaDest) — captures the net change in each account's balance, giving the model a direct signal of money movement
Large transaction flag — marks transactions above the 99.9th percentile amount threshold, derived from the training set to avoid data leakage
Merchant destination flag — identifies whether the destination account is a merchant (prefix M), since merchant accounts behave differently from customer accounts


Predictive Modeling
Two gradient boosting models were trained and evaluated using a time-based train/validation split (last 20% of time steps held out for validation).
Class Imbalance Handling:

Inverse-frequency sample weights applied during training
Macro F1-score used as the primary evaluation metric to equally weight all urgency levels

CatBoost (jupyter_notebook_files/model_train.ipynb)

Native handling of categorical features (no encoding needed)
Early stopping with 200-iteration patience
Tuned depth, L2 regularization, and learning rate

XGBoost (python_files/xgtry.py)

One-hot encoded categorical features
Per-row sample weights for imbalance handling
Tuned max depth, subsampling, and regularization

ModelValidation Macro F1CatBoostTBDXGBoostTBD

Final Conclusion
To be updated after final model experiments are complete.
The project demonstrates that gradient boosting models with careful feature engineering and class imbalance handling are well-suited for fraud urgency classification. Balance discrepancy flags and delta features proved to be particularly informative signals.

How to Run the Project
bash# Clone the repo
git clone https://github.com/adamsilver2005/Fraud-Detection-Classification.git
cd Fraud-Detection-Classification

# Install dependencies
pip install -r requirements.txt

# Run EDA
python python_files/data_fraud.py

# Run XGBoost model
python python_files/xgtry.py
Place train.csv and test.csv inside the data/ folder before running.
Notebooks should be run in order:

data_preprocessing.ipynb — engineers new features and saves enriched CSVs to outputs/
model_train.ipynb — trains CatBoost on the enriched data and outputs a submission file


Attribution
This project originated as a team submission for HackML 2026 on Kaggle, built collaboratively with Peyton289, TrentB159, and ArwinSepahram.
This repository is my personal continuation of that work with independent improvements and refactoring.
Competition: FRAUD | HackML 2026