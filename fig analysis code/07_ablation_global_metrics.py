from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mutual_info_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


REAL_DATA_PATH = Path(r"D:\LLM generate data\data\CHARLS_processed_2020.csv")
DATASET_PATHS = {
    "Full Model": Path(r"D:\LLM generate data\data\synthetic_data_v12.csv"),
    "Ablation A": Path(r"D:\LLM generate data\data\synthetic_data_v16.csv"),
    "Ablation B": Path(r"D:\LLM generate data\data\synthetic_data_v17bnew.csv"),
    "Ablation C": Path(r"D:\LLM generate data\data\synthetic_data_v18.csv"),
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

MISSING_LABEL = "__MISSING__"


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
    df1 = pd.read_csv(path)
    #df1 = df1.dropna()
    return df1


def align_columns(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    missing = [col for col in FEATURE_ORDER if col not in df.columns]
    if missing:
        raise ValueError(f"{dataset_name} is missing required columns: {missing}")
    return df[FEATURE_ORDER].copy()


def categorical_labels(series: pd.Series) -> pd.Series:
    return series.astype("object").where(series.notna(), MISSING_LABEL).astype(str)


def categorical_frame(df: pd.DataFrame) -> pd.DataFrame:
    return df.astype("object").where(df.notna(), MISSING_LABEL).astype(str)


def compute_mi_matrix(df: pd.DataFrame) -> pd.DataFrame:
    matrix = pd.DataFrame(0.0, index=FEATURE_ORDER, columns=FEATURE_ORDER)

    for i, col_i in enumerate(FEATURE_ORDER):
        for j, col_j in enumerate(FEATURE_ORDER):
            if i >= j:
                continue

            mi_value = mutual_info_score(
                categorical_labels(df[col_i]),
                categorical_labels(df[col_j]),
            )
            matrix.loc[col_i, col_j] = mi_value
            matrix.loc[col_j, col_i] = mi_value

    return matrix


def upper_triangle_values(matrix: pd.DataFrame) -> np.ndarray:
    upper_idx = np.triu_indices_from(matrix.to_numpy(), k=1)
    return matrix.to_numpy()[upper_idx]


def mi_distance_metrics(real_matrix: pd.DataFrame, synthetic_matrix: pd.DataFrame) -> dict[str, float]:
    real_values = upper_triangle_values(real_matrix)
    synthetic_values = upper_triangle_values(synthetic_matrix)
    diff = synthetic_values - real_values

    if np.std(real_values) == 0 or np.std(synthetic_values) == 0:
        correlation = np.nan
    else:
        correlation = float(np.corrcoef(real_values, synthetic_values)[0, 1])

    return {
        "MI Mean": float(np.mean(synthetic_values)),
        "MI Std": float(np.std(synthetic_values, ddof=0)),
        "MAE": float(np.mean(np.abs(diff))),
        "RMSE": float(np.sqrt(np.mean(diff**2))),
        "Correlation": correlation,
    }


def balanced_pair(
    real_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    n = min(len(real_df), len(synthetic_df))
    real_sample = real_df.sample(n=n, random_state=random_state).reset_index(drop=True)
    synthetic_sample = synthetic_df.sample(n=n, random_state=random_state).reset_index(drop=True)
    return real_sample, synthetic_sample


def encode_pair(real_sample: pd.DataFrame, synthetic_sample: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    real_labeled = real_sample.copy()
    real_labeled["is_synthetic"] = 0

    synthetic_labeled = synthetic_sample.copy()
    synthetic_labeled["is_synthetic"] = 1

    combined = pd.concat([real_labeled, synthetic_labeled], ignore_index=True)
    y = combined.pop("is_synthetic").to_numpy()
    encoded = pd.get_dummies(categorical_frame(combined), columns=FEATURE_ORDER, prefix_sep="=", dtype=np.float32)
    return encoded.to_numpy(dtype=np.float32), y


def propensity_mse(probabilities: np.ndarray, base_rate: float) -> float:
    return float(np.mean((probabilities - base_rate) ** 2))


def discriminator_metrics(
    real_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    random_state: int,
    test_size: float,
    dt_max_depth: int,
    dt_min_samples_leaf: int,
) -> dict[str, float]:
    real_sample, synthetic_sample = balanced_pair(real_df, synthetic_df, random_state)
    x, y = encode_pair(real_sample, synthetic_sample)

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

    dt = DecisionTreeClassifier(
        max_depth=dt_max_depth,
        min_samples_leaf=dt_min_samples_leaf,
        random_state=random_state,
    )
    dt.fit(x_train, y_train)
    dt_prob = dt.predict_proba(x_test)[:, 1]

    return {
        "LR AUC": float(roc_auc_score(y_test, lr_prob)),
        "LR PMSE": propensity_mse(lr_prob, float(y_test.mean())),
        "DT AUC": float(roc_auc_score(y_test, dt_prob)),
        "DT PMSE": propensity_mse(dt_prob, float(y_test.mean())),
    }


def format_float(value: float, digits: int = 4) -> str:
    if np.isnan(value):
        return "nan"
    return f"{value:.{digits}f}".rstrip("0").rstrip(".")


def build_output_table(results: dict[str, dict[str, float]]) -> pd.DataFrame:
    rows = []
    for dataset, metrics in results.items():
        rows.append(
            {
                "Dataset": dataset,
                "MI Mean": format_float(metrics["MI Mean"]),
                "MI Std": format_float(metrics["MI Std"]),
                "MAE": format_float(metrics["MAE"]),
                "RMSE": format_float(metrics["RMSE"]),
                "Correlation": format_float(metrics["Correlation"]),
                "LR AUC": format_float(metrics["LR AUC"]),
                "LR PMSE": format_float(metrics["LR PMSE"]),
                "DT AUC": format_float(metrics["DT AUC"]),
                "DT PMSE": format_float(metrics["DT PMSE"]),
            }
        )
    return pd.DataFrame(rows)


def print_table(table: pd.DataFrame) -> None:
    pd.set_option("display.max_colwidth", 80)
    pd.set_option("display.width", 180)
    print()
    print("TABLE VIII".center(120))
    print("GLOBAL METRICS FOR ABLATION STUDY".center(120))
    print()
    print(table.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print MI-distance and discriminator metrics for ablation synthetic datasets."
    )
    parser.add_argument("--real-csv", type=Path, default=REAL_DATA_PATH, help="Path to CHARLS_processed_2020.csv.")
    parser.add_argument("--full-csv", type=Path, default=DATASET_PATHS["Full Model"], help="Path to full-model CSV.")
    parser.add_argument("--ablation-a-csv", type=Path, default=DATASET_PATHS["Ablation A"], help="Path to ablation A CSV.")
    parser.add_argument("--ablation-b-csv", type=Path, default=DATASET_PATHS["Ablation B"], help="Path to ablation B CSV.")
    parser.add_argument("--ablation-c-csv", type=Path, default=DATASET_PATHS["Ablation C"], help="Path to ablation C CSV.")
    parser.add_argument("--test-size", type=float, default=0.3, help="Held-out fraction for discriminator evaluation.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for discriminator sampling and split.")
    parser.add_argument("--dt-max-depth", type=int, default=5, help="Decision tree maximum depth.")
    parser.add_argument("--dt-min-samples-leaf", type=int, default=50, help="Decision tree minimum samples per leaf.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    real_df = align_columns(load_real_data(args.real_csv), "Real")
    datasets = {
        "Full Model": align_columns(load_synthetic_data(args.full_csv), "Full Model"),
        "Ablation A": align_columns(load_synthetic_data(args.ablation_a_csv), "Ablation A"),
        "Ablation B": align_columns(load_synthetic_data(args.ablation_b_csv), "Ablation B"),
        "Ablation C": align_columns(load_synthetic_data(args.ablation_c_csv), "Ablation C"),
    }

    real_matrix = compute_mi_matrix(real_df)
    results = {}
    for idx, (name, df) in enumerate(datasets.items()):
        synthetic_matrix = compute_mi_matrix(df)
        metrics = mi_distance_metrics(real_matrix, synthetic_matrix)
        metrics.update(
            discriminator_metrics(
                real_df=real_df,
                synthetic_df=df,
                random_state=args.random_state + idx * 100,
                test_size=args.test_size,
                dt_max_depth=args.dt_max_depth,
                dt_min_samples_leaf=args.dt_min_samples_leaf,
            )
        )
        results[name] = metrics

    print_table(build_output_table(results))


if __name__ == "__main__":
    main()
