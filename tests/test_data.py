from pathlib import Path

import numpy as np
import pandas as pd

from src.data import (
    label_to_binary,
    load_binary_dataset,
    make_feature_frame,
    normalize_column_name,
    profile_schemas,
)


FIXTURES = Path(__file__).parent / "fixtures"


def test_normalize_column_name():
    assert normalize_column_name(" Flow Byts/s ") == "flow_byts_s"
    assert normalize_column_name("CWE Flag Count") == "cwe_flag_count"


def test_label_mapping_keeps_only_benign_and_ddos():
    assert label_to_binary("Benign") == 0
    assert label_to_binary("DDoS attacks-LOIC-HTTP") == 1
    assert label_to_binary("DDOS attack-HOIC") == 1
    assert label_to_binary("DDOS attack-LOIC-UDP") == 1
    assert label_to_binary("DoS attacks-Hulk") is None
    assert label_to_binary("Label") is None


def test_schema_profile_handles_different_source_file_columns():
    schemas = profile_schemas(FIXTURES)
    assert schemas["02-20-2018.csv"][0] == "flow_id"
    assert "flow_id" not in schemas["02-21-2018.csv"]


def test_load_binary_dataset_filters_non_scope_labels():
    frame, profile = load_binary_dataset(
        FIXTURES, max_rows_per_class_per_file=10, chunksize=2
    )
    assert set(frame["target"]) == {0, 1}
    assert "SSH-Bruteforce" not in set(frame["raw_label"])
    assert profile["rows_loaded"] == 7


def test_make_feature_frame_drops_identifiers_and_dst_port():
    frame = pd.DataFrame(
        {
            "flow_id": ["abc"],
            "src_ip": ["10.0.0.1"],
            "dst_port": [80],
            "flow_byts_s": [np.inf],
            "target": [1],
            "label": ["DDOS attack-HOIC"],
        }
    )
    X, columns = make_feature_frame(frame)
    assert "flow_id" not in columns
    assert "src_ip" not in columns
    assert "dst_port" not in columns
    assert "flow_byts_s" not in columns
    assert X.empty


def test_make_feature_frame_can_keep_destination_port_for_shortcut_analysis():
    frame = pd.DataFrame(
        {
            "dst_port": [80, 443],
            "flow_byts_s": [1200.0, 1500.0],
            "target": [0, 1],
        }
    )
    X, columns = make_feature_frame(frame, drop_dst_port=False)
    assert columns == ["dst_port", "flow_byts_s"]
    assert X["dst_port"].tolist() == [80, 443]
