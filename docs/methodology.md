# Methodology

## Research Question

Can a small supervised machine-learning pipeline distinguish benign traffic from DDoS network flows in CSE-CIC-IDS2018 without relying on obvious leakage or random-row test splits?

## Corrected Scope

The final scope is binary classification:

```text
Benign vs DDoS
```

Included attack labels:

- `DDoS attacks-LOIC-HTTP`
- `DDOS attack-HOIC`
- `DDOS attack-LOIC-UDP`

Excluded labels include brute force, DoS, botnet, web attack, SQL injection, and infiltration labels because they are outside the DDoS-only question.

## Data Split

The final test set is grouped by source CSV file:

- Train/validation source: `02-20-2018.csv`
- Final test source: `02-21-2018.csv`

This is stricter than a random row split. It tests whether patterns learned from one DDoS capture day transfer to a different DDoS capture day. The result is less flattering but more honest.

## Leakage Controls

The pipeline removes or avoids:

- Flow identifiers
- Source and destination IP addresses
- Source port
- Timestamp
- Destination port by default
- Source filename as a model feature

`02-20-2018.csv` has extra identifier columns, so schema alignment is handled explicitly.

All learned preprocessing is inside scikit-learn pipelines and is fitted on training data only.

## Missing and Infinite Values

The original notebook dropped every row with any missing value. That removed too much data and allowed missing-value handling to influence the research scope.

The repaired version:

- Converts numeric features safely
- Replaces infinite values with missing values
- Removes all-empty feature columns
- Uses median imputation fitted only on training data

## Models

The project compares:

- Majority baseline
- Decision Tree
- Random Forest

SMOTE is not used. A class-weighted tree baseline is simpler and avoids oversampling leakage risk. MLP is not retained because the original notebook did not converge and it did not add value to the corrected, scoped experiment.

## Metrics

Reported metrics:

- Precision
- Recall
- F1-score
- Balanced accuracy
- ROC-AUC
- PR-AUC
- False-positive rate
- Confusion matrix
- Training time
- Inference time

Classification thresholds for real models are selected on validation data only and then applied once to the final holdout.

## Main Finding

Validation performance on the `02-20-2018.csv` split is extremely high, but final testing on `02-21-2018.csv` performs poorly for recall. This suggests the model is learning capture-day or attack-variant artifacts that do not transfer cleanly to the next DDoS day.

That finding replaces the old notebook's misleading near-perfect result.
