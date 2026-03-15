# Project Context — BNP Business Case

## Project Structure

```
notebooks/
  1)Data_Understanding.ipynb
  2a)Data_Preparation.ipynb          ← builds the ABT
  3a)Data_Preprocessing_Models_Churn.ipynb
  3a)Data_Preprocessing_Models_Settlement.ipynb
src/code/
  data_preparation.py               ← reusable cleaning functions
```

---

## Two Objectives

| Objective | Notebook | Target Variable | Question |
|-----------|----------|-----------------|----------|
| **Churn** | `3a)...Churn` | `IS_CHURN` | Will the customer come back after their contract ends? |
| **Settlement** | `3a)...Settlement` | `IS_EARLY_SETTLER` | Will the customer end their contract early (SAN vs SOL)? |

---

## ABT — Data Basis (notebook 2a)

### Raw BDOSS grain
- 2,658,187 rows = monthly observation snapshots per contract
- 185,600 unique contracts (CONTRIB × DOSSIER)
- 148,729 unique customers (CONTRIB)

### Aggregation (Steps 1–3 in 2a)
- **Step 1**: Contract-level POS flags (HAS_SOL, HAS_SAN, HAS_RBT per contract)
- **Step 2**: Contract-level numeric aggregation (1 row per CONTRIB × DOSSIER)
- **Step 3**: Customer-level aggregation (1 row per CONTRIB) — collapses all contracts

**No snapshot date / time boundary is used.** Features are aggregated over the customer's ENTIRE history, including post-settlement observations. Some post-settlement leaky features are dropped in 3a to compensate.

### Target Variable Definitions

**IS_EARLY_SETTLER** (Settlement model):
```python
IS_EARLY_SETTLER = (EVER_SAN == 1) | (EVER_RBT == 1)
# 1 = early/self-initiated settlement (SAN or RBT)
# 0 = on-time settlement (SOL)
```

**IS_CHURN** (Churn model):
```python
last_end   = max(LAST_OBS_DATE_SOL, LAST_OBS_DATE_SAN, LAST_OBS_DATE_RBT)
gap_days   = LAST_DCREAT - last_end  (days)
IS_CHURN   = had_end_event AND (LAST_DCREAT <= last_end OR gap_days > 30)
# 1 = settled and did NOT open a new contract within 30 days → churned
# 0 = never settled OR opened new contract within 30 days → retained
```

**IS_CHURN distribution in full ABT:**
- IS_CHURN=0: 82,407 (55.4%)
- IS_CHURN=1: 66,322 (44.6%)

**POS status values:**
- `ENC` = active/ongoing contract
- `SOL` = settled on time (customer-driven, on schedule)
- `SAN` = early settlement (client-initiated)
- `RBT` = settled with reimbursement (SCLI > 0)

---

## Training Population — Both Models

```python
mask_pay = client_data[client_data[["EVER_SAN", "EVER_SOL", "EVER_RBT"]].eq(1).any(axis=1)]
# 67,440 customers — those with at least one settlement event
```

### IS_CHURN distribution within mask_pay (training set):
- IS_CHURN=1: 66,322 (98.34%) ← extreme imbalance
- IS_CHURN=0:  1,118  (1.66%)

The 98%/2% split is mostly real business behavior (most customers don't return after settling), amplified by the strict 30-day threshold. SMOTE is applied during CV to handle this.


---

**Important:** The models themselves (Sections 1–9) are valid and unaffected. Section 10 is an add-on scoring step — the core ML pipeline uses mask_pay correctly throughout.


---

## Model Performance Summary

| Model | Target | Val ROC-AUC | Notes |
|-------|--------|-------------|-------|
| HistGBM (tuned) | IS_CHURN | 0.9092 | Train-val gap 6.9% ⚠️, Renewal F1=0.58 |
| HistGBM (tuned) | IS_EARLY_SETTLER | 0.9858 | No overfitting, well-calibrated |

Both use: 5-fold stratified CV, SMOTE, RobustScaler, OHE for categoricals, 58 features after selection.

---

## Key Architectural Decision

A **snapshot-date / contract-level ABT** was discussed as the proper long-term fix (features computed up to a reference date per customer, preventing post-settlement data from leaking into features). This was **not implemented** because it would increase ML complexity (within-customer correlation, CV strategy, etc.). The current customer-level ABT is accepted with the scoring fix above as the pragmatic solution.
