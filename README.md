# FRAUD_HackML_2026

# Dataset Description

Each row represents a single mobile money transaction generated from the **PaySim** simulator. All identifiers are **synthetic**. The target variable for this competition is **`urgency_level`**.

---

## Files

- **`train.csv`** — Training set  
- **`test.csv`** — Test set  
- **`sample_submission.csv`** — Sample submission file in the correct format

---

## Columns

### `id`
- **Type:** Integer  
- **Description:** Unique transaction identifier assigned during dataset preparation. Used to align predictions with the correct rows in `test.csv`.

### `step`
- **Type:** Integer  
- **Description:** Time step of the transaction, where **1 step = 1 hour** since the start of the simulation.

### `type`
- **Type:** Categorical (string)  
- **Description:** Type of transaction. Common values include:
  - `CASH_IN`
  - `CASH_OUT`
  - `DEBIT`
  - `PAYMENT`
  - `TRANSFER`

### `amount`
- **Type:** Numeric (float)  
- **Description:** Transaction amount in local currency.

### `oldbalanceOrg`
- **Type:** Numeric (float)  
- **Description:** Balance of the origin account before the transaction.

### `newbalanceOrg`
- **Type:** Numeric (float)  
- **Description:** Balance of the origin account after the transaction.

### `oldbalanceDest`
- **Type:** Numeric (float)  
- **Description:** Balance of the destination account before the transaction. Some destination accounts may represent merchants (in the original PaySim formulation), for which certain balance fields may not apply.

### `newbalanceDest`
- **Type:** Numeric (float)  
- **Description:** Balance of the destination account after the transaction.

### `nameOrig`
- **Type:** String  
- **Description:** The customer initiating the transaction.

### `nameDest`
- **Type:** String  
- **Description:** The transaction’s recipient customer.

---

## Target Variable: `urgency_level`

- **Type:** Integer (categorical)  
- **Valid values:** `{0, 1, 2, 3}`  
- **Description:** A derived categorical label indicating the recommended urgency of fraud investigation for a transaction. Higher values represent higher risk and require faster response.

### Class Meanings

- **0 — No Action:** Transaction appears legitimate; no investigation required.  
- **1 — Monitor:** Low-risk suspicious behavior; monitor for patterns or repeated activity.  
- **2 — Review:** Likely fraudulent; should be reviewed by an analyst.  
- **3 — Immediate Action:** High-risk fraud; requires urgent investigation or intervention.

---

## Notes

- `urgency_level` is organizer-defined for this competition and is **not** part of the original PaySim dataset.
- The exact derivation procedure is intentionally not disclosed to participants.
- The dataset is **highly imbalanced**, reflecting real-world fraud rarity.
