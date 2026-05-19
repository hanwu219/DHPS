from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


REAL_DATA_PATH = Path(r"D:\LLM generate data\data\CHARLS_processed_2020.csv")
MODEL_PATHS = {
    "Ours": Path(r"D:\LLM generate data\data\synthetic_data_v12.csv"),
    "Copula": Path(r"D:\LLM generate data\data\synthetic_data_copula.csv"),
    "TVAE": Path(r"D:\LLM generate data\data\synthetic_data_tvae.csv"),
    "XGB": Path(r"D:\LLM generate data\data\synthetic_data_xgb.csv"),
    "TabPFN": Path(r"D:\LLM generate data\data\synthetic_data_tabpfn.csv"),
    "GReaT": Path(r"D:\LLM generate data\data\synthetic_data_great.csv"),
}

FEATURE_ORDER = [
    "age_bin",
    "gender",
    "marry",
    "edu",
    "income_bin",
    "family_size",
    "health_status",
    "hospital",
    "exercise",
    "ins",
    "satlife",
    "social_need",
]

ATTRIBUTE_TARGETS = ["income_bin", "health_status", "hospital", "satlife", "social_need"]

DISPLAY_NAMES = {
    "age_bin": "Age",
    "gender": "Gender",
    "marry": "Marry",
    "edu": "Edu",
    "income_bin": "Income",
    "family_size": "Family",
    "health_status": "Health",
    "hospital": "Hospital",
    "exercise": "Exercise",
    "ins": "Ins",
    "satlife": "Satlife",
    "social_need": "Social",
}


def load_real_data(path: Path) -> pd.DataFrame:
    """Load and preprocess the real CHARLS data with the fixed rules."""
    df = pd.read_csv(path)

    df.drop(columns=["iwy", "row_id"], inplace=True, errors="ignore")
    df = df[df["income_total"] >= 0].copy()
    df["age_bin"] = pd.cut(
        df["age"],
        bins=[0, 59, 64, 69, 74, 79, np.inf],
        labels=["60-", "60-64", "65-69", "70-74", "75-79", "80+"],
    )
    df["income_bin"] = pd.qcut(
        df["income_total"],
        q=3,
        labels=["Low", "Medium", "High"],
        duplicates="drop",
    )

    df = df.drop(columns=["age", "income_total"])
    df = df[df["family_size"] <= 10]
    df = df.dropna()
    return df


def load_synthetic_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.dropna()
    return df


def align_columns(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    missing = [col for col in FEATURE_ORDER if col not in df.columns]
    if missing:
        raise ValueError(f"{dataset_name} is missing required columns: {missing}")
    return df[FEATURE_ORDER].copy()


def stringify(df: pd.DataFrame) -> pd.DataFrame:
    return df[FEATURE_ORDER].astype("object").astype(str)


def encode_frames(frames: list[pd.DataFrame], columns: list[str]) -> list[np.ndarray]:
    encoded_columns = [[] for _ in frames]

    for col in columns:
        combined = pd.concat([frame[col].astype("object").astype(str) for frame in frames], ignore_index=True)
        codes, _ = pd.factorize(combined, sort=True)
        start = 0
        for idx, frame in enumerate(frames):
            end = start + len(frame)
            encoded_columns[idx].append(codes[start:end])
            start = end

    arrays = []
    for columns_for_frame in encoded_columns:
        arrays.append(np.column_stack(columns_for_frame).astype(np.int32))
    return arrays


def nearest_hamming_distances(
    query: np.ndarray,
    reference: np.ndarray,
    feature_indices: np.ndarray | None = None,
    chunk_size: int = 200,
) -> np.ndarray:
    if feature_indices is not None:
        query = query[:, feature_indices]
        reference = reference[:, feature_indices]

    distances = np.empty(len(query), dtype=float)
    for start in range(0, len(query), chunk_size):
        end = min(start + chunk_size, len(query))
        mismatch_count = (query[start:end, None, :] != reference[None, :, :]).sum(axis=2)
        distances[start:end] = mismatch_count.min(axis=1) / query.shape[1]
    return distances


def count_exact_copies(real_df: pd.DataFrame, synthetic_df: pd.DataFrame) -> int:
    real_records = set(map(tuple, stringify(real_df).to_numpy()))
    synthetic_records = map(tuple, stringify(synthetic_df).to_numpy())
    return sum(record in real_records for record in synthetic_records)


def sample_frame(df: pd.DataFrame, n: int, random_state: int) -> pd.DataFrame:
    sample_n = min(n, len(df))
    return df.sample(n=sample_n, random_state=random_state).reset_index(drop=True)


def generate_marginal_nonmembers(real_df: pd.DataFrame, n: int, random_state: int) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    sampled = {}
    for col in FEATURE_ORDER:
        values = real_df[col].dropna().to_numpy()
        sampled[col] = rng.choice(values, size=n, replace=True)
    return pd.DataFrame(sampled, columns=FEATURE_ORDER)


def risk_label(auc: float) -> str:
    if auc >= 0.75:
        return "High"
    if auc >= 0.65:
        return "Moderate"
    return "Low"


def evaluate_dcr(
    real_sample: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    chunk_size: int,
) -> dict[str, object]:
    """Compute DCR from every synthetic record to a fixed sampled real reference set."""
    real_encoded, synthetic_encoded = encode_frames([real_sample, synthetic_df], FEATURE_ORDER)
    dcr = nearest_hamming_distances(synthetic_encoded, real_encoded, chunk_size=chunk_size)
    copy_num = count_exact_copies(real_sample, synthetic_df)
    copy_pct = copy_num / len(synthetic_df) * 100.0 if len(synthetic_df) else 0.0

    return {
        "Sample": len(real_sample),
        "DCR Mean": float(np.mean(dcr)),
        "DCR 5th Percentile": float(np.percentile(dcr, 5)),
        "Copy Num": copy_num,
        "Copy Percent": copy_pct,
    }


def evaluate_membership_inference(
    real_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    attack_sample_size: int,
    reference_size: int,
    random_state: int,
    chunk_size: int,
) -> dict[str, object]:
    member_sample = sample_frame(real_df, attack_sample_size, random_state)
    nonmember_sample = generate_marginal_nonmembers(real_df, len(member_sample), random_state + 17)
    synthetic_reference = sample_frame(synthetic_df, reference_size, random_state + 31)

    member_encoded, nonmember_encoded, reference_encoded = encode_frames(
        [member_sample, nonmember_sample, synthetic_reference],
        FEATURE_ORDER,
    )
    member_dcr = nearest_hamming_distances(member_encoded, reference_encoded, chunk_size=chunk_size)
    nonmember_dcr = nearest_hamming_distances(nonmember_encoded, reference_encoded, chunk_size=chunk_size)

    labels = np.concatenate([np.ones(len(member_dcr)), np.zeros(len(nonmember_dcr))])
    scores = -np.concatenate([member_dcr, nonmember_dcr])
    auc = float(roc_auc_score(labels, scores))
    advantage = max(0.0, 2.0 * (auc - 0.5))

    return {
        "Attack Sample": len(member_sample),
        "MIA AUC": auc,
        "Advantage": advantage,
        "Member DCR": float(np.mean(member_dcr)),
        "Nonmember DCR": float(np.mean(nonmember_dcr)),
        "Risk": risk_label(auc),
    }


def majority_vote(values: np.ndarray) -> np.ndarray:
    predictions = []
    max_code = int(values.max()) if values.size else 0
    for row in values:
        counts = np.bincount(row, minlength=max_code + 1)
        predictions.append(int(np.argmax(counts)))
    return np.asarray(predictions, dtype=np.int32)


def attribute_attack_accuracy(
    real_sample: pd.DataFrame,
    synthetic_reference: pd.DataFrame,
    target_col: str,
    k_neighbors: int,
    chunk_size: int,
) -> float:
    real_encoded, synthetic_encoded = encode_frames([real_sample, synthetic_reference], FEATURE_ORDER)
    target_idx = FEATURE_ORDER.index(target_col)
    feature_indices = np.array([idx for idx in range(len(FEATURE_ORDER)) if idx != target_idx])

    predictions = np.empty(len(real_encoded), dtype=np.int32)
    for start in range(0, len(real_encoded), chunk_size):
        end = min(start + chunk_size, len(real_encoded))
        query = real_encoded[start:end, :][:, feature_indices]
        reference = synthetic_encoded[:, feature_indices]
        mismatch_count = (query[:, None, :] != reference[None, :, :]).sum(axis=2)
        k = min(k_neighbors, len(synthetic_encoded))
        nearest_idx = np.argpartition(mismatch_count, kth=k - 1, axis=1)[:, :k]
        neighbor_targets = synthetic_encoded[nearest_idx, target_idx]
        predictions[start:end] = majority_vote(neighbor_targets)

    truth = real_encoded[:, target_idx]
    return float(np.mean(predictions == truth))


def evaluate_attribute_inference(
    real_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    attack_sample_size: int,
    reference_size: int,
    k_neighbors: int,
    random_state: int,
    chunk_size: int,
) -> dict[str, object]:
    real_sample = sample_frame(real_df, attack_sample_size, random_state)
    synthetic_reference = sample_frame(synthetic_df, reference_size, random_state + 53)

    rows = []
    for target_col in ATTRIBUTE_TARGETS:
        attack_acc = attribute_attack_accuracy(
            real_sample,
            synthetic_reference,
            target_col=target_col,
            k_neighbors=k_neighbors,
            chunk_size=chunk_size,
        )
        baseline_acc = float(real_df[target_col].astype("object").astype(str).value_counts(normalize=True).iloc[0])
        rows.append(
            {
                "target": target_col,
                "attack_accuracy": attack_acc,
                "baseline_accuracy": baseline_acc,
                "gain": attack_acc - baseline_acc,
            }
        )

    details = pd.DataFrame(rows)
    worst = details.sort_values("gain", ascending=False).iloc[0]
    return {
        "Attack Sample": len(real_sample),
        "AI Accuracy": float(details["attack_accuracy"].mean()),
        "Baseline": float(details["baseline_accuracy"].mean()),
        "AI Gain": float(details["gain"].mean()),
        "Worst Attribute": DISPLAY_NAMES.get(str(worst["target"]), str(worst["target"])),
    }


def format_float(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def build_dcr_table(results: dict[str, dict[str, object]]) -> pd.DataFrame:
    rows = []
    for model, row in results.items():
        rows.append(
            {
                "Model": model,
                "Sample": row["Sample"],
                "DCR Mean": format_float(float(row["DCR Mean"])),
                "DCR 5th Percentile": format_float(float(row["DCR 5th Percentile"])),
                "Copy Num": f"{row['Copy Num']} ({float(row['Copy Percent']):.1f}%)",
            }
        )
    return pd.DataFrame(rows)


def build_membership_table(results: dict[str, dict[str, object]]) -> pd.DataFrame:
    rows = []
    for model, row in results.items():
        rows.append(
            {
                "Model": model,
                "Sample": row["Attack Sample"],
                "MIA AUC": format_float(float(row["MIA AUC"])),
                "Advantage": format_float(float(row["Advantage"])),
                "Member DCR": format_float(float(row["Member DCR"])),
                "Nonmember DCR": format_float(float(row["Nonmember DCR"])),
                "Risk": row["Risk"],
            }
        )
    return pd.DataFrame(rows)


def build_attribute_table(results: dict[str, dict[str, object]]) -> pd.DataFrame:
    rows = []
    for model, row in results.items():
        rows.append(
            {
                "Model": model,
                "Sample": row["Attack Sample"],
                "AI Accuracy": format_float(float(row["AI Accuracy"])),
                "Baseline": format_float(float(row["Baseline"])),
                "AI Gain": format_float(float(row["AI Gain"])),
                "Worst Attribute": row["Worst Attribute"],
            }
        )
    return pd.DataFrame(rows)


def print_table(title: str, table: pd.DataFrame) -> None:
    pd.set_option("display.max_colwidth", 80)
    pd.set_option("display.width", 180)
    print()
    print(title.center(120))
    print()
    print(table.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print DCR, membership inference, and attribute inference privacy risk tables."
    )
    parser.add_argument("--real-csv", type=Path, default=REAL_DATA_PATH, help="Path to CHARLS_processed_2020.csv.")
    parser.add_argument("--ours-csv", type=Path, default=MODEL_PATHS["Ours"], help="Path to our synthetic CSV.")
    parser.add_argument("--copula-csv", type=Path, default=MODEL_PATHS["Copula"], help="Path to Copula CSV.")
    parser.add_argument("--tvae-csv", type=Path, default=MODEL_PATHS["TVAE"], help="Path to TVAE CSV.")
    parser.add_argument("--xgb-csv", type=Path, default=MODEL_PATHS["XGB"], help="Path to XGB CSV.")
    parser.add_argument("--tabpfn-csv", type=Path, default=MODEL_PATHS["TabPFN"], help="Path to TabPFN CSV.")
    parser.add_argument("--great-csv", type=Path, default=MODEL_PATHS["GReaT"], help="Path to GReaT CSV.")
    parser.add_argument("--dcr-sample-size", type=int, default=200, help="Real records sampled as the DCR reference set.")
    parser.add_argument("--dcr-random-state", type=int, default=24, help="Random seed for df_real.sample in DCR.")
    parser.add_argument("--attack-sample-size", type=int, default=1000, help="Candidate records for attack evaluation.")
    parser.add_argument("--reference-size", type=int, default=3000, help="Synthetic records used as attack reference.")
    parser.add_argument("--k-neighbors", type=int, default=5, help="Nearest neighbors used for attribute inference.")
    parser.add_argument("--chunk-size", type=int, default=200, help="Chunk size for nearest-neighbor distance computation.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    real_df = align_columns(load_real_data(args.real_csv), "Real")
    synthetic_datasets = {
        "Ours": align_columns(load_synthetic_data(args.ours_csv), "Ours"),
        "Copula": align_columns(load_synthetic_data(args.copula_csv), "Copula"),
        "TVAE": align_columns(load_synthetic_data(args.tvae_csv), "TVAE"),
        "XGB": align_columns(load_synthetic_data(args.xgb_csv), "XGB"),
        "TabPFN": align_columns(load_synthetic_data(args.tabpfn_csv), "TabPFN"),
        "GReaT": align_columns(load_synthetic_data(args.great_csv), "GReaT"),
    }
    dcr_real_sample = sample_frame(real_df, args.dcr_sample_size, args.dcr_random_state)

    dcr_results = {}
    membership_results = {}
    attribute_results = {}
    for idx, (model, synthetic_df) in enumerate(synthetic_datasets.items()):
        seed = args.random_state + idx * 100
        dcr_results[model] = evaluate_dcr(
            dcr_real_sample,
            synthetic_df,
            chunk_size=args.chunk_size,
        )
        membership_results[model] = evaluate_membership_inference(
            real_df,
            synthetic_df,
            attack_sample_size=args.attack_sample_size,
            reference_size=args.reference_size,
            random_state=seed,
            chunk_size=args.chunk_size,
        )
        attribute_results[model] = evaluate_attribute_inference(
            real_df,
            synthetic_df,
            attack_sample_size=args.attack_sample_size,
            reference_size=args.reference_size,
            k_neighbors=args.k_neighbors,
            random_state=seed,
            chunk_size=args.chunk_size,
        )

    print_table("TABLE VII\nDCR VERIFICATION QUANTITATIVE COMPARISON TABLE", build_dcr_table(dcr_results))
    print_table("TABLE VIII\nMEMBERSHIP INFERENCE QUANTITATIVE COMPARISON TABLE", build_membership_table(membership_results))
    print_table("TABLE IX\nATTRIBUTE INFERENCE QUANTITATIVE COMPARISON TABLE", build_attribute_table(attribute_results))


if __name__ == "__main__":
    main()
