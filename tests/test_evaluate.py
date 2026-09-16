from src.evaluate import best_f1_threshold, compute_binary_metrics


def test_compute_binary_metrics_uses_positive_class_scores():
    metrics = compute_binary_metrics(
        y_true=[0, 0, 1, 1],
        y_pred=[0, 1, 1, 1],
        y_score=[0.05, 0.60, 0.80, 0.95],
    )
    assert metrics["confusion_matrix"] == {"tn": 1, "fp": 1, "fn": 0, "tp": 2}
    assert metrics["false_positive_rate"] == 0.5
    assert metrics["roc_auc"] == 1.0
    assert metrics["pr_auc"] == 1.0


def test_best_f1_threshold_returns_validation_cutoff():
    threshold = best_f1_threshold(
        y_true=[0, 0, 1, 1],
        y_score=[0.05, 0.20, 0.65, 0.90],
    )
    assert 0.2 < threshold <= 0.65
