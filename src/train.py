from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

try:
    from .data import RANDOM_SEED, load_binary_dataset, make_feature_frame
    from .evaluate import compute_binary_metrics, positive_class_probability
except ImportError:  # Allows `python src/train.py` from the repo root.
    from data import RANDOM_SEED, load_binary_dataset, make_feature_frame
    from evaluate import compute_binary_metrics, positive_class_probability


def build_models(seed: int) -> dict[str, Pipeline]:
    return {
        "majority_baseline": Pipeline(
            [("imputer", SimpleImputer(strategy="median")), ("model", DummyClassifier(strategy="most_frequent"))]
        ),
        "decision_tree": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    DecisionTreeClassifier(
                        max_depth=10,
                        min_samples_leaf=25,
                        class_weight="balanced",
                        random_state=seed,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=120,
                        max_depth=16,
                        min_samples_leaf=10,
                        class_weight="balanced_subsample",
                        n_jobs=-1,
                        random_state=seed,
                    ),
                ),
            ]
        ),
    }


def evaluate_model(name: str, model: Pipeline, X, y) -> dict:
    start = time.perf_counter()
    predictions = model.predict(X)
    scores = positive_class_probability(model, X)
    inference_seconds = time.perf_counter() - start
    metrics = compute_binary_metrics(y, predictions, scores)
    metrics["inference_seconds"] = float(inference_seconds)
    metrics["rows"] = int(len(y))
    metrics["model"] = name
    return metrics


def save_confusion_matrix(metrics: dict, path: Path) -> None:
    cm = metrics["confusion_matrix"]
    matrix = [[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]]
    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1], labels=["Benign", "DDoS"])
    ax.set_yticks([0, 1], labels=["Benign", "DDoS"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    for row in range(2):
        for col in range(2):
            ax.text(col, row, matrix[row][col], ha="center", va="center")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run_experiment(args: argparse.Namespace) -> dict:
    reports_dir = Path(args.reports_dir)
    figures_dir = reports_dir / "figures"
    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    data, data_profile = load_binary_dataset(
        args.data_dir,
        max_rows_per_class_per_file=args.max_rows_per_class_per_file,
        chunksize=args.chunksize,
        random_state=args.seed,
    )
    if args.holdout_file not in set(data["source_file"]):
        raise ValueError(f"Holdout file {args.holdout_file!r} was not loaded.")

    test_frame = data[data["source_file"] == args.holdout_file].copy()
    train_pool = data[data["source_file"] != args.holdout_file].copy()
    if test_frame["target"].nunique() < 2:
        raise ValueError("Final holdout must contain both Benign and DDoS rows.")
    if train_pool["target"].nunique() < 2:
        raise ValueError("Training pool must contain both Benign and DDoS rows.")

    train_frame, val_frame = train_test_split(
        train_pool,
        test_size=args.validation_size,
        stratify=train_pool["target"],
        random_state=args.seed,
    )

    X_train, feature_columns = make_feature_frame(
        train_frame, drop_dst_port=not args.keep_dst_port
    )
    X_val, _ = make_feature_frame(
        val_frame, feature_columns=feature_columns, drop_dst_port=not args.keep_dst_port
    )
    X_test, _ = make_feature_frame(
        test_frame, feature_columns=feature_columns, drop_dst_port=not args.keep_dst_port
    )
    y_train = train_frame["target"].astype(int)
    y_val = val_frame["target"].astype(int)
    y_test = test_frame["target"].astype(int)

    model_results: dict[str, dict] = {}
    fitted_models: dict[str, Pipeline] = {}
    for name, model in build_models(args.seed).items():
        start = time.perf_counter()
        model.fit(X_train, y_train)
        train_seconds = time.perf_counter() - start
        fitted_models[name] = model
        model_results[name] = {
            "train_seconds": float(train_seconds),
            "validation": evaluate_model(name, model, X_val, y_val),
        }

    best_model_name = max(
        model_results,
        key=lambda name: model_results[name]["validation"]["pr_auc"],
    )
    best_model = fitted_models[best_model_name]
    for name, model in fitted_models.items():
        model_results[name]["test"] = evaluate_model(name, model, X_test, y_test)

    best_test_metrics = model_results[best_model_name]["test"]
    save_confusion_matrix(best_test_metrics, figures_dir / "confusion_matrix.png")

    importance_rows = []
    if not args.skip_permutation_importance:
        subset_size = min(args.permutation_rows, len(X_test))
        X_perm = X_test.sample(n=subset_size, random_state=args.seed)
        y_perm = y_test.loc[X_perm.index]
        result = permutation_importance(
            best_model,
            X_perm,
            y_perm,
            n_repeats=5,
            random_state=args.seed,
            scoring="average_precision",
            n_jobs=-1,
        )
        importance = pd.DataFrame(
            {
                "feature": feature_columns,
                "importance_mean": result.importances_mean,
                "importance_std": result.importances_std,
            }
        ).sort_values("importance_mean", ascending=False)
        importance.to_csv(reports_dir / "permutation_importance.csv", index=False)
        importance_rows = importance.head(10).to_dict(orient="records")

    payload = {
        "dataset": "CSE-CIC-IDS2018 on AWS",
        "scope": "Binary classification of Benign vs DDoS network flows",
        "seed": args.seed,
        "holdout_file": args.holdout_file,
        "drop_dst_port": not args.keep_dst_port,
        "dropped_identifier_features": [
            "flow_id",
            "src_ip",
            "src_port",
            "dst_ip",
            "timestamp",
            "dst_port",
        ],
        "data_profile": data_profile,
        "split_counts": {
            "train": train_frame["target"].value_counts().sort_index().to_dict(),
            "validation": val_frame["target"].value_counts().sort_index().to_dict(),
            "test": test_frame["target"].value_counts().sort_index().to_dict(),
        },
        "raw_label_counts_by_split": {
            "train": train_frame["raw_label"].value_counts().to_dict(),
            "validation": val_frame["raw_label"].value_counts().to_dict(),
            "test": test_frame["raw_label"].value_counts().to_dict(),
        },
        "feature_count": len(feature_columns),
        "feature_columns": feature_columns,
        "model_results": model_results,
        "best_model": best_model_name,
        "top_permutation_importance": importance_rows,
        "methodology_notes": [
            "All preprocessing is fitted inside sklearn Pipelines on training data only.",
            "SMOTE is intentionally not used; class-weighted tree models are simpler and avoid oversampling leakage risk.",
            "MLP is intentionally not retained because the legacy notebook did not converge and tree models are sufficient for this scoped tabular baseline.",
            "Final test data is grouped by source CSV file rather than random row split.",
        ],
    }

    output_path = reports_dir / "metrics.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    rows = []
    for model_name, values in model_results.items():
        for split_name in ("validation", "test"):
            row = {"model": model_name, "split": split_name}
            row.update(values[split_name])
            row.pop("confusion_matrix")
            rows.append(row)
    pd.DataFrame(rows).to_csv(reports_dir / "model_results.csv", index=False)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Benign-vs-DDoS detector.")
    parser.add_argument("--data-dir", default="data", help="Directory containing CSE-CIC-IDS2018 CSV files.")
    parser.add_argument("--reports-dir", default="reports", help="Output directory for metrics and figures.")
    parser.add_argument("--holdout-file", default="02-21-2018.csv", help="CSV file reserved for final testing.")
    parser.add_argument("--max-rows-per-class-per-file", type=int, default=10_000)
    parser.add_argument("--chunksize", type=int, default=100_000)
    parser.add_argument("--validation-size", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--keep-dst-port", action="store_true", help="Keep destination port for shortcut analysis.")
    parser.add_argument("--skip-permutation-importance", action="store_true")
    parser.add_argument("--permutation-rows", type=int, default=5_000)
    return parser.parse_args()


def main() -> None:
    payload = run_experiment(parse_args())
    best = payload["best_model"]
    test = payload["model_results"][best]["test"]
    print(f"Best model: {best}")
    print(
        "Final test: "
        f"F1={test['f1']:.4f}, PR-AUC={test['pr_auc']:.4f}, "
        f"ROC-AUC={test['roc_auc']:.4f}, FPR={test['false_positive_rate']:.4f}"
    )


if __name__ == "__main__":
    main()
