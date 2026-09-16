from argparse import Namespace

import pandas as pd

from src.train import run_experiment


def test_run_experiment_reports_destination_port_when_enabled(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    def write_day(filename, attack_label):
        rows = []
        for index in range(8):
            rows.append(
                {
                    "Dst Port": 80,
                    "Protocol": 6,
                    "Flow Duration": 1000 + index,
                    "Flow Byts/s": 1200.0 + index,
                    "Label": "Benign",
                }
            )
            rows.append(
                {
                    "Dst Port": 443,
                    "Protocol": 6,
                    "Flow Duration": 500 + index,
                    "Flow Byts/s": 100000.0 + index,
                    "Label": attack_label,
                }
            )
        pd.DataFrame(rows).to_csv(data_dir / filename, index=False)

    write_day("02-20-2018.csv", "DDoS attacks-LOIC-HTTP")
    write_day("02-21-2018.csv", "DDOS attack-HOIC")
    payload = run_experiment(
        Namespace(
            data_dir=data_dir,
            reports_dir=tmp_path / "reports",
            holdout_file="02-21-2018.csv",
            include_files=["02-20-2018.csv", "02-21-2018.csv"],
            max_rows_per_class_per_file=8,
            chunksize=100,
            validation_size=0.25,
            seed=42,
            keep_dst_port=True,
            skip_permutation_importance=True,
            permutation_rows=10,
        )
    )

    assert payload["drop_dst_port"] is False
    assert "dst_port" in payload["feature_columns"]
    assert "dst_port" not in payload["dropped_identifier_features"]
