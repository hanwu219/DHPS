from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


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


def balanced_pair(
    real_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    n = min(len(real_df), len(synthetic_df))
    if n == 0:
        raise ValueError("At least one dataset has no rows after preprocessing.")

    real_sample = real_df.sample(n=n, random_state=random_state).reset_index(drop=True)
    synthetic_sample = synthetic_df.sample(n=n, random_state=random_state).reset_index(drop=True)
    return real_sample, synthetic_sample


def encode_pair(real_sample: pd.DataFrame, synthetic_sample: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    labeled_real = real_sample.copy()
    labeled_real["is_synthetic"] = 0

    labeled_synthetic = synthetic_sample.copy()
    labeled_synthetic["is_synthetic"] = 1

    combined = pd.concat([labeled_real, labeled_synthetic], ignore_index=True)
    y = combined.pop("is_synthetic").to_numpy()

    categorical = combined.astype("object").astype(str)
    encoded = pd.get_dummies(
        categorical,
        columns=FEATURE_ORDER,
        prefix_sep="=",
        dtype=np.float32,
    )
    return encoded.to_numpy(dtype=np.float32), y, encoded.columns.tolist()


def propensity_mse(probabilities: np.ndarray, base_rate: float) -> float:
    return float(np.mean((probabilities - base_rate) ** 2))


def source_feature(encoded_feature: str) -> str:
    return encoded_feature.split("=", maxsplit=1)[0]


def aggregate_feature_scores(
    feature_names: list[str],
    lr_scores: np.ndarray,
    dt_scores: np.ndarray,
) -> dict[str, float]:
    def normalize(scores: np.ndarray) -> np.ndarray:
        total = float(np.sum(scores))
        if total <= 0:
            return np.zeros_like(scores, dtype=float)
        return scores / total

    combined_scores = normalize(lr_scores) + normalize(dt_scores)
    aggregated = {feature: 0.0 for feature in FEATURE_ORDER}
    for encoded_name, score in zip(feature_names, combined_scores):
        source = source_feature(encoded_name)
        aggregated[source] = aggregated.get(source, 0.0) + float(score)
    return aggregated


def leakage_features(
    aggregated_scores: dict[str, float],
    lr_auc: float,
    dt_auc: float,
    threshold: float,
    top_k: int,
) -> str:
    discriminator_strength = max(abs(lr_auc - 0.5), abs(dt_auc - 0.5))
    if discriminator_strength < threshold:
        return "/"

    top_features = sorted(aggregated_scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
    return "/".join(DISPLAY_NAMES.get(feature, feature) for feature, score in top_features if score > 0)


def evaluate_model(
    real_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    random_state: int,
    test_size: float,
    dt_max_depth: int,
    dt_min_samples_leaf: int,
    leakage_threshold: float,
    leakage_top_k: int,
) -> dict[str, object]:
    real_sample, synthetic_sample = balanced_pair(real_df, synthetic_df, random_state)
    x, y, feature_names = encode_pair(real_sample, synthetic_sample)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    lr = LogisticRegression(max_iter=2000, solver="lbfgs")
    lr.fit(x_train, y_train)
    lr_prob = lr.predict_proba(x_test)[:, 1]
    lr_auc = float(roc_auc_score(y_test, lr_prob))
    lr_pmse = propensity_mse(lr_prob, float(y_test.mean()))

    dt = DecisionTreeClassifier(
        max_depth=dt_max_depth,
        min_samples_leaf=dt_min_samples_leaf,
        random_state=random_state,
    )
    dt.fit(x_train, y_train)
    dt_prob = dt.predict_proba(x_test)[:, 1]
    dt_auc = float(roc_auc_score(y_test, dt_prob))
    dt_pmse = propensity_mse(dt_prob, float(y_test.mean()))

    feature_scores = aggregate_feature_scores(
        feature_names,
        np.abs(lr.coef_[0]),
        dt.feature_importances_,
    )

    return {
        "LR AUC": lr_auc,
        "LR PMSE": lr_pmse,
        "DT AUC": dt_auc,
        "DT PMSE": dt_pmse,
        "Discriminator Strength": max(abs(lr_auc - 0.5), abs(dt_auc - 0.5)),
        "Leakage Features": leakage_features(
            feature_scores,
            lr_auc,
            dt_auc,
            threshold=leakage_threshold,
            top_k=leakage_top_k,
        ),
        "Rows": len(real_sample),
    }


def best_models(
    metrics: dict[str, dict[str, object]],
    column: str,
    tolerance: float,
    auc_metric: bool,
) -> str:
    if auc_metric:
        scores = {name: abs(float(row[column]) - 0.5) for name, row in metrics.items()}
    else:
        scores = {name: float(row[column]) for name, row in metrics.items()}

    best_score = min(scores.values())
    winners = [name for name, score in scores.items() if score <= best_score + tolerance]
    return "/".join(winners)


def best_leakage_models(metrics: dict[str, dict[str, object]]) -> str:
    no_leakage = [name for name, row in metrics.items() if row["Leakage Features"] == "/"]
    if no_leakage:
        return "/".join(no_leakage)

    strengths = {name: float(row["Discriminator Strength"]) for name, row in metrics.items()}
    best_strength = min(strengths.values())
    return "/".join(name for name, strength in strengths.items() if strength <= best_strength + 0.002)


def format_metric(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return str(value)


def build_output_table(
    metrics: dict[str, dict[str, object]],
    best_tolerance: float,
) -> pd.DataFrame:
    rows = []
    for name, row in metrics.items():
        rows.append(
            {
                "Metrics": name,
                "LR AUC": format_metric(row["LR AUC"]),
                "LR PMSE": format_metric(row["LR PMSE"]),
                "DT AUC": format_metric(row["DT AUC"]),
                "DT PMSE": format_metric(row["DT PMSE"]),
                "Leakage Features": row["Leakage Features"],
            }
        )

    rows.append(
        {
            "Metrics": "BEST",
            "LR AUC": best_models(metrics, "LR AUC", best_tolerance, auc_metric=True),
            "LR PMSE": best_models(metrics, "LR PMSE", best_tolerance, auc_metric=False),
            "DT AUC": best_models(metrics, "DT AUC", best_tolerance, auc_metric=True),
            "DT PMSE": best_models(metrics, "DT PMSE", best_tolerance, auc_metric=False),
            "Leakage Features": best_leakage_models(metrics),
        }
    )
    return pd.DataFrame(rows)


def print_table(table: pd.DataFrame) -> None:
    pd.set_option("display.max_colwidth", 80)
    pd.set_option("display.width", 180)
    print()
    print("TABLE V".center(120))
    print("QUANTITATIVE COMPARISON TABLE FOR ADVERSARIAL VERIFICATION".center(120))
    print()
    print(table.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Use simple discriminators to distinguish real and synthetic CHARLS data."
    )
    parser.add_argument("--real-csv", type=Path, default=REAL_DATA_PATH, help="Path to CHARLS_processed_2020.csv.")
    parser.add_argument("--ours-csv", type=Path, default=MODEL_PATHS["Ours"], help="Path to our synthetic CSV.")
    parser.add_argument("--copula-csv", type=Path, default=MODEL_PATHS["Copula"], help="Path to Copula CSV.")
    parser.add_argument("--tvae-csv", type=Path, default=MODEL_PATHS["TVAE"], help="Path to TVAE CSV.")
    parser.add_argument("--xgb-csv", type=Path, default=MODEL_PATHS["XGB"], help="Path to XGB CSV.")
    parser.add_argument("--tabpfn-csv", type=Path, default=MODEL_PATHS["TabPFN"], help="Path to TabPFN CSV.")
    parser.add_argument("--great-csv", type=Path, default=MODEL_PATHS["GReaT"], help="Path to GReaT CSV.")
    parser.add_argument("--test-size", type=float, default=0.3, help="Held-out test fraction.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for sampling and splitting.")
    parser.add_argument("--dt-max-depth", type=int, default=5, help="Decision tree maximum depth.")
    parser.add_argument("--dt-min-samples-leaf", type=int, default=50, help="Decision tree minimum samples per leaf.")
    parser.add_argument(
        "--leakage-threshold",
        type=float,
        default=0.08,
        help="Show leakage features when max AUC distance from 0.5 reaches this threshold.",
    )
    parser.add_argument("--leakage-top-k", type=int, default=3, help="Number of top leakage features to print.")
    parser.add_argument("--best-tolerance", type=float, default=0.002, help="Tolerance for ties in the BEST row.")
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

    metrics = {}
    for name, synthetic_df in synthetic_datasets.items():
        metrics[name] = evaluate_model(
            real_df=real_df,
            synthetic_df=synthetic_df,
            random_state=args.random_state,
            test_size=args.test_size,
            dt_max_depth=args.dt_max_depth,
            dt_min_samples_leaf=args.dt_min_samples_leaf,
            leakage_threshold=args.leakage_threshold,
            leakage_top_k=args.leakage_top_k,
        )

    table = build_output_table(metrics, best_tolerance=args.best_tolerance)
    print_table(table)


if __name__ == "__main__":
    main()
