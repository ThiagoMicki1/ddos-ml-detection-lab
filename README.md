# DDoS ML Detection Lab

A repaired machine-learning portfolio project for DDoS detection using CSE-CIC-IDS2018 network-flow CSV data.

The original notebook reported extremely high accuracy, but it used an easy binary setup, broad row dropping, random row splitting, and leakage-prone preprocessing. I rebuilt the project to make the methodology more honest: source-file holdout, explicit DDoS scope, safer preprocessing, reproducible tests, and clear limitations.

## Security Problem

DDoS detection is a common network-security use case: defenders want to separate normal traffic from traffic patterns that may indicate service disruption attempts. This project does not build a production IDS. It is a local research lab showing how easy it is for ML intrusion-detection experiments to look great when the split is too easy.

## Dataset

Dataset: CSE-CIC-IDS2018 on AWS.

Official sources:

- https://www.unb.ca/cic/datasets/ids-2018.html
- https://registry.opendata.aws/cse-cic-ids2018/

Raw CSV files and `archive.zip` are not committed. See [docs/dataset.md](docs/dataset.md).

## Experimental Design

Scope:

```text
Binary classification: Benign vs DDoS
```

Included DDoS labels:

- `DDoS attacks-LOIC-HTTP`
- `DDOS attack-HOIC`
- `DDOS attack-LOIC-UDP`

Final split:

- Train/validation: `02-20-2018.csv`
- Final test: `02-21-2018.csv`

This tests cross-day and cross-variant generalization instead of random-row memorization.

## Leakage Controls

The repaired pipeline drops or avoids:

- Flow ID
- Source IP
- Destination IP
- Source port
- Timestamp
- Destination port by default
- Source filename as a feature

All imputation is fitted inside scikit-learn pipelines on training data only.

## Models Evaluated

| Model | Purpose |
|---|---|
| Majority baseline | Sanity check |
| Decision Tree | Simple interpretable tree model |
| Random Forest | Stronger tabular baseline |

SMOTE and MLP are intentionally not used in the final version. SMOTE was a leakage risk in the original workflow, and the previous MLP did not converge.

## Corrected Results

Final run:

```bash
python -m src.train --max-rows-per-class-per-file 10000 --reports-dir reports
```

Best model selected by validation PR-AUC: `random_forest`.

| Model | Split | Precision | Recall | F1 | Balanced Accuracy | ROC-AUC | PR-AUC | FPR |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Majority baseline | Test | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.5000 | 0.5257 | 0.0000 |
| Decision Tree | Test | 0.5634 | 0.0040 | 0.0079 | 0.5003 | 0.6670 | 0.6817 | 0.0034 |
| Random Forest | Validation | 0.9967 | 0.9993 | 0.9980 | 0.9987 | 1.0000 | 1.0000 | 0.0020 |
| Random Forest | Test | 0.7500 | 0.0006 | 0.0012 | 0.5002 | 0.5947 | 0.7589 | 0.0002 |

The important result is not that the model is strong. It is that the model looks excellent on validation data from the same source day, then fails to generalize to the held-out DDoS day. That is a useful security-engineering lesson.

Generated artifacts:

- [reports/metrics.json](reports/metrics.json)
- [reports/model_results.csv](reports/model_results.csv)
- [reports/permutation_importance.csv](reports/permutation_importance.csv)
- [reports/figures/confusion_matrix.png](reports/figures/confusion_matrix.png)

## Repository Structure

```text
.
├── README.md
├── requirements.txt
├── notebooks/
│   └── ddos_detection_analysis.ipynb
├── src/
│   ├── data.py
│   ├── evaluate.py
│   └── train.py
├── tests/
├── reports/
│   ├── figures/
│   ├── metrics.json
│   └── model_results.csv
├── docs/
│   ├── dataset.md
│   └── methodology.md
└── .github/workflows/ci.yml
```

## Reproduce Locally

Create an environment:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

Run tests:

```bash
python -m pytest -q
```

Run the corrected experiment:

```bash
python -m src.train --max-rows-per-class-per-file 10000 --reports-dir reports
```

Run the notebook:

```bash
jupyter nbconvert --to notebook --execute notebooks/ddos_detection_analysis.ipynb --output executed_ddos_detection_analysis.ipynb
```

## What I Changed From the Original Notebook

- Replaced random row splitting with source-file holdout.
- Removed identifier and shortcut features.
- Removed `dropna()` as a broad cleaning strategy.
- Fixed AUC handling instead of hiding failures behind broad exceptions.
- Removed SMOTE-before-cross-validation leakage.
- Removed non-converged MLP from the final model set.
- Replaced "Untitled.ipynb" with a clean, executable notebook.
- Added tests, CI, dependency file, docs, and reproducible reports.

## Limitations

- This is still a sampled local experiment, not a full-scale training job.
- The source-file holdout is stricter than the old random split, but it is still within one public dataset.
- The model performs poorly on DDoS recall for the held-out day.
- The dataset may contain capture artifacts that do not represent real enterprise traffic.
- This project should not be used as a production detector.

## Ethical Use

This repository is for defensive security education and research review. It does not include attack tooling, packet replay instructions, or deployment guidance for offensive activity.

## Future Improvements

- Add cross-dataset testing against another public flow dataset.
- Compare with a time-aware split if reliable timestamps are normalized.
- Add calibration analysis and threshold curves.

## Interview Talking Points

- Why random row splits can exaggerate IDS performance.
- Why source-file holdout is a better test for capture-day generalization.
- Why a model with high validation scores can still fail on security-relevant holdout data.
- Why leakage-prone features like IPs, timestamps, and ports need careful treatment.
