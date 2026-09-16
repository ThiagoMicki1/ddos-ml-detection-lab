from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


RANDOM_SEED = 42
DDOS_LABELS = {
    "ddos attacks-loic-http",
    "ddos attack-hoic",
    "ddos attack-loic-udp",
}
DROP_COLUMNS = {
    "flow_id",
    "src_ip",
    "src_port",
    "dst_ip",
    "timestamp",
    "label",
    "raw_label",
    "source_file",
    "target",
}
DEFAULT_FEATURE_COLUMNS = {
    "dst_port",
    "protocol",
    "flow_duration",
    "tot_fwd_pkts",
    "tot_bwd_pkts",
    "totlen_fwd_pkts",
    "totlen_bwd_pkts",
    "fwd_pkt_len_mean",
    "fwd_pkt_len_std",
    "bwd_pkt_len_mean",
    "bwd_pkt_len_std",
    "flow_byts_s",
    "flow_pkts_s",
    "flow_iat_mean",
    "flow_iat_std",
    "flow_iat_max",
    "flow_iat_min",
    "fwd_pkts_s",
    "bwd_pkts_s",
    "pkt_len_mean",
    "pkt_len_std",
    "pkt_len_var",
    "syn_flag_cnt",
    "ack_flag_cnt",
    "init_fwd_win_byts",
    "init_bwd_win_byts",
    "active_mean",
    "idle_mean",
}


def normalize_column_name(name: object) -> str:
    text = str(name).strip().lower()
    text = re.sub(r"[^0-9a-z]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def normalize_columns(columns: Iterable[object]) -> list[str]:
    return [normalize_column_name(column) for column in columns]


def label_to_binary(label: object) -> int | None:
    cleaned = str(label).strip().lower()
    if cleaned == "benign":
        return 0
    if cleaned in DDOS_LABELS:
        return 1
    return None


def list_csv_files(data_dir: str | Path) -> list[Path]:
    return sorted(Path(data_dir).glob("*.csv"))


def select_csv_files(
    data_dir: str | Path, include_files: Iterable[str] | None = None
) -> list[Path]:
    files = list_csv_files(data_dir)
    if include_files is None:
        return files
    wanted = set(include_files)
    selected = [path for path in files if path.name in wanted]
    missing = sorted(wanted - {path.name for path in selected})
    if missing:
        raise FileNotFoundError(f"Missing expected CSV files: {', '.join(missing)}")
    return selected


def profile_schemas(data_dir: str | Path) -> dict[str, list[str]]:
    schemas: dict[str, list[str]] = {}
    for path in list_csv_files(data_dir):
        header = pd.read_csv(path, nrows=0, low_memory=False)
        schemas[path.name] = normalize_columns(header.columns)
    return schemas


def load_binary_dataset(
    data_dir: str | Path,
    *,
    max_rows_per_class_per_file: int = 10_000,
    chunksize: int = 100_000,
    random_state: int = RANDOM_SEED,
    feature_columns: set[str] | None = None,
    include_files: Iterable[str] | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Load a deterministic sample for benign-vs-DDoS classification.

    The full CSE-CIC-IDS2018 CSVs are large. This loader scans labels in chunks,
    keeps only Benign and DDoS variants, and samples a bounded number of rows per
    class per source file so the experiment is reproducible on a laptop.
    """
    frames: list[pd.DataFrame] = []
    label_counts_before: dict[str, Counter] = {}
    label_counts_after: dict[str, Counter] = {}
    schemas = profile_schemas(data_dir)
    per_chunk_cap = max(50, max_rows_per_class_per_file // 10)
    selected_columns = set(feature_columns or DEFAULT_FEATURE_COLUMNS)
    selected_columns.add("label")

    files = select_csv_files(data_dir, include_files)
    for file_index, path in enumerate(files):
        before = Counter()
        after = Counter()
        for chunk_index, chunk in enumerate(
            pd.read_csv(
                path,
                chunksize=chunksize,
                low_memory=False,
                usecols=lambda column: normalize_column_name(column) in selected_columns,
            )
        ):
            chunk.columns = normalize_columns(chunk.columns)
            if "label" not in chunk.columns:
                continue

            raw_labels = chunk["label"].astype(str).str.strip()
            before.update(raw_labels.value_counts(dropna=False).to_dict())
            mapped = raw_labels.map(label_to_binary)
            keep = mapped.notna()
            if not keep.any():
                continue

            kept = chunk.loc[keep].copy()
            kept["raw_label"] = raw_labels.loc[keep].to_numpy()
            kept["target"] = mapped.loc[keep].astype(int).to_numpy()
            kept["source_file"] = path.name

            sampled_parts = []
            for target_value, part in kept.groupby("target", sort=False):
                take = min(len(part), per_chunk_cap)
                sampled = part.sample(
                    n=take,
                    random_state=random_state + file_index * 1000 + chunk_index,
                )
                sampled_parts.append(sampled)
                after.update(sampled["raw_label"].value_counts().to_dict())
            frames.extend(sampled_parts)

        label_counts_before[path.name] = before
        label_counts_after[path.name] = after

    if not frames:
        raise ValueError(f"No Benign or DDoS rows found under {data_dir}")

    sampled = pd.concat(frames, ignore_index=True)
    bounded_groups = []
    for _, group in sampled.groupby(["source_file", "target"], sort=False):
        bounded_groups.append(
            group.sample(
                n=min(len(group), max_rows_per_class_per_file),
                random_state=random_state,
            )
        )
    sampled = pd.concat(bounded_groups, ignore_index=True)

    profile = {
        "schemas": {name: len(cols) for name, cols in schemas.items()},
        "label_counts_before": {
            name: dict(counts) for name, counts in label_counts_before.items()
        },
        "sampled_label_counts": {
            name: dict(counts) for name, counts in label_counts_after.items()
        },
        "rows_loaded": int(len(sampled)),
        "max_rows_per_class_per_file": max_rows_per_class_per_file,
        "candidate_feature_columns": sorted(selected_columns - {"label"}),
        "included_files": [path.name for path in files],
    }
    return sampled, profile


def make_feature_frame(
    frame: pd.DataFrame,
    *,
    feature_columns: list[str] | None = None,
    drop_dst_port: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    drop_columns = set(DROP_COLUMNS)
    if drop_dst_port:
        drop_columns.add("dst_port")

    candidate = frame.drop(
        columns=[column for column in drop_columns if column in frame.columns],
        errors="ignore",
    )
    numeric = candidate.apply(pd.to_numeric, errors="coerce")
    numeric = numeric.replace([np.inf, -np.inf], np.nan)

    if feature_columns is None:
        feature_columns = [
            column for column in numeric.columns if numeric[column].notna().any()
        ]

    return numeric.reindex(columns=feature_columns), feature_columns
