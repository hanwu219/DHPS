from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


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

TARGET_COLUMNS = FEATURE_ORDER.copy()
CORE_TARGETS = ["income_bin", "family_size", "edu", "satlife", "social_need"]

DISPLAY_NAMES = {
    "age_bin": "Age",
    "gender": "Gender",
    "marry": "Marry",
    "edu": "Edu",
    "income_bin": "Income",
    "family_size": "Family_size",
    "health_status": "Health",
    "hospital": "Hospital",
    "exercise": "Exercise",
    "ins": "Ins",
    "satlife": "Satlife",
    "social_need": "Social_need",
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


def as_categorical_strings(df: pd.DataFrame) -> pd.DataFrame:
    return df.astype("object").astype(str)


def make_classifier(feature_columns: list[str], random_state: int) -> Pipeline:
    encoder = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                feature_columns,
            )
        ],
        remainder="drop",
    )
    return Pipeline(
        steps=[
            ("encoder", encoder),
            (
                "classifier",
                LogisticRegression(max_iter=2000, solver="lbfgs", random_state=random_state),
            ),
        ]
    )


def fit_predict(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
    random_state: int,
) -> np.ndarray:
    feature_cols = [col for col in FEATURE_ORDER if col != target_col]
    x_train = as_categorical_strings(train_df[feature_cols])
    y_train = train_df[target_col].astype("object").astype(str)
    x_test = as_categorical_strings(test_df[feature_cols])

    if y_train.nunique() < 2:
        classifier = DummyClassifier(strategy="most_frequent")
        classifier.fit(x_train, y_train)
        return classifier.predict(x_test)

    classifier = make_classifier(feature_cols, random_state=random_state)
    classifier.fit(x_train, y_train)
    return classifier.predict(x_test)


def evaluate_f1(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
    average: str,
    random_state: int,
) -> float:
    y_test = test_df[target_col].astype("object").astype(str)
    y_pred = fit_predict(train_df, test_df, target_col, random_state=random_state)
    return float(f1_score(y_test, y_pred, average=average, zero_division=0))


def sample_synthetic_train(
    synthetic_df: pd.DataFrame,
    train_size: int,
    random_state: int,
) -> pd.DataFrame:
    if len(synthetic_df) >= train_size:
        return synthetic_df.sample(n=train_size, random_state=random_state).reset_index(drop=True)
    return synthetic_df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)


def evaluate_tstr_model(
    real_baseline_f1: dict[str, float],
    real_train: pd.DataFrame,
    real_test: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    average: str,
    random_state: int,
) -> tuple[dict[str, float], pd.DataFrame]:
    synthetic_train = sample_synthetic_train(
        synthetic_df,
        train_size=len(real_train),
        random_state=random_state,
    )

    rows = []
    for idx, target_col in enumerate(TARGET_COLUMNS):
        tstr_f1 = evaluate_f1(
            train_df=synthetic_train,
            test_df=real_test,
            target_col=target_col,
            average=average,
            random_state=random_state + idx,
        )
        baseline_f1 = real_baseline_f1[target_col]
        f1_gap = abs(tstr_f1 - baseline_f1)
        f1_loss = baseline_f1 - tstr_f1
        recovery = np.nan if baseline_f1 == 0 else (tstr_f1 / baseline_f1) * 100.0
        rows.append(
            {
                "target": target_col,
                "real_baseline_f1": baseline_f1,
                "tstr_f1": tstr_f1,
                "f1_gap": f1_gap,
                "f1_loss": f1_loss,
                "recovery": recovery,
            }
        )

    details = pd.DataFrame(rows)
    summary = {
        "mean_f1_gap": float(details["f1_gap"].mean()),
        "mean_recovery": float(details["recovery"].mean()),
    }
    return summary, details


def core_feature_consistency(details: pd.DataFrame, gap_threshold: float) -> str:
    core = details[details["target"].isin(CORE_TARGETS)]
    count = int((core["f1_gap"] <= gap_threshold).sum())
    return f"{count} / {len(CORE_TARGETS)}"


def worst_collapse_type(details: pd.DataFrame, top_k: int) -> str:
    ranked = details.sort_values("f1_loss", ascending=False)
    positive_losses = ranked[ranked["f1_loss"] > 0]
    if positive_losses.empty:
        positive_losses = ranked

    names = [
        DISPLAY_NAMES.get(target, target)
        for target in positive_losses.head(top_k)["target"].tolist()
    ]
    return " / ".join(names)


def build_output_table(metrics: dict[str, dict[str, object]]) -> pd.DataFrame:
    rows = []
    for model, row in metrics.items():
        rows.append(
            {
                "Model": model,
                "Mean F1 Gap": f"{float(row['Mean F1 Gap']):.3f}".rstrip("0").rstrip("."),
                "Mean Recovery": f"{float(row['Mean Recovery']):.1f}%",
                "Core Feature Consistency": row["Core Feature Consistency"],
                "Worst Collapse Type": row["Worst Collapse Type"],
            }
        )
    return pd.DataFrame(rows)


def print_table(table: pd.DataFrame) -> None:
    pd.set_option("display.max_colwidth", 90)
    pd.set_option("display.width", 180)
    print()
    print("TABLE VI".center(120))
    print("TSTR QUANTITATIVE COMPARISON TABLE".center(120))
    print()
    print(table.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train on synthetic data and test on real data for multiple synthetic CHARLS models."
    )
    parser.add_argument("--real-csv", type=Path, default=REAL_DATA_PATH, help="Path to CHARLS_processed_2020.csv.")
    parser.add_argument("--ours-csv", type=Path, default=MODEL_PATHS["Ours"], help="Path to our synthetic CSV.")
    parser.add_argument("--copula-csv", type=Path, default=MODEL_PATHS["Copula"], help="Path to Copula CSV.")
    parser.add_argument("--tvae-csv", type=Path, default=MODEL_PATHS["TVAE"], help="Path to TVAE CSV.")
    parser.add_argument("--xgb-csv", type=Path, default=MODEL_PATHS["XGB"], help="Path to XGB CSV.")
    parser.add_argument("--tabpfn-csv", type=Path, default=MODEL_PATHS["TabPFN"], help="Path to TabPFN CSV.")
    parser.add_argument("--great-csv", type=Path, default=MODEL_PATHS["GReaT"], help="Path to GReaT CSV.")
    parser.add_argument("--test-size", type=float, default=0.3, help="Held-out real test fraction.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for splitting and sampling.")
    parser.add_argument(
        "--average",
        choices=["weighted", "macro", "micro"],
        default="weighted",
        help="F1 averaging method for each categorical prediction task.",
    )
    parser.add_argument(
        "--core-gap-threshold",
        type=float,
        default=0.02,
        help="A core feature is consistent if absolute F1 gap is at or below this value.",
    )
    parser.add_argument("--worst-top-k", type=int, default=3, help="Number of worst-collapse target columns to print.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    real_df = align_columns(load_real_data(args.real_csv), "Real")
    real_train, real_test = train_test_split(
        real_df,
        test_size=args.test_size,
        random_state=args.random_state,
        shuffle=True,
    )
    real_train = real_train.reset_index(drop=True)
    real_test = real_test.reset_index(drop=True)

    real_baseline_f1 = {}
    for idx, target_col in enumerate(TARGET_COLUMNS):
        real_baseline_f1[target_col] = evaluate_f1(
            train_df=real_train,
            test_df=real_test,
            target_col=target_col,
            average=args.average,
            random_state=args.random_state + idx,
        )

    synthetic_datasets = {
        "Ours": align_columns(load_synthetic_data(args.ours_csv), "Ours"),
        "Copula": align_columns(load_synthetic_data(args.copula_csv), "Copula"),
        "TVAE": align_columns(load_synthetic_data(args.tvae_csv), "TVAE"),
        "XGB": align_columns(load_synthetic_data(args.xgb_csv), "XGB"),
        "TabPFN": align_columns(load_synthetic_data(args.tabpfn_csv), "TabPFN"),
        "GReaT": align_columns(load_synthetic_data(args.great_csv), "GReaT"),
    }

    metrics = {}
    for model_idx, (model, synthetic_df) in enumerate(synthetic_datasets.items()):
        summary, details = evaluate_tstr_model(
            real_baseline_f1=real_baseline_f1,
            real_train=real_train,
            real_test=real_test,
            synthetic_df=synthetic_df,
            average=args.average,
            random_state=args.random_state + model_idx * 100,
        )
        metrics[model] = {
            "Mean F1 Gap": summary["mean_f1_gap"],
            "Mean Recovery": summary["mean_recovery"],
            "Core Feature Consistency": core_feature_consistency(details, args.core_gap_threshold),
            "Worst Collapse Type": worst_collapse_type(details, args.worst_top_k),
            "Worst Loss": float(details["f1_loss"].max()),
        }

    table = build_output_table(metrics)
    print_table(table)


if __name__ == "__main__":
    main()
