from __future__ import annotations

import argparse
import os
import warnings
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
warnings.filterwarnings("ignore", message="Could not find the number of physical cores.*")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE


REAL_DATA_PATH = Path(r"D:\LLM generate data\data\CHARLS_processed_2020.csv")
SYNTHETIC_DATA_PATH = Path(r"D:\LLM generate data\data\synthetic_data_v11.csv")
OUTPUT_DIR = Path(__file__).resolve().parent / "tsne_outputs"

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
    df = pd.read_csv(path)
    df = df.dropna()
    return df

def align_columns(real_df: pd.DataFrame, synthetic_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    missing_real = [col for col in FEATURE_ORDER if col not in real_df.columns]
    missing_synthetic = [col for col in FEATURE_ORDER if col not in synthetic_df.columns]
    if missing_real or missing_synthetic:
        raise ValueError(
            "Column mismatch. "
            f"Missing in real data: {missing_real}; "
            f"missing in synthetic data: {missing_synthetic}"
        )

    return real_df[FEATURE_ORDER].copy(), synthetic_df[FEATURE_ORDER].copy()


def sample_data(
    real_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    sample_size: int,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    real_n = min(sample_size, len(real_df))
    synthetic_n = min(sample_size, len(synthetic_df))

    real_sample = real_df.sample(n=real_n, random_state=random_state).reset_index(drop=True)
    synthetic_sample = synthetic_df.sample(n=synthetic_n, random_state=random_state).reset_index(drop=True)
    return real_sample, synthetic_sample


def encode_for_tsne(real_sample: pd.DataFrame, synthetic_sample: pd.DataFrame) -> tuple[np.ndarray, pd.Series]:
    labeled_real = real_sample.copy()
    labeled_real["dataset_source"] = "Real Data"

    labeled_synthetic = synthetic_sample.copy()
    labeled_synthetic["dataset_source"] = "Synthetic Data"

    combined = pd.concat([labeled_real, labeled_synthetic], ignore_index=True)
    labels = combined.pop("dataset_source")

    categorical = combined.astype("object").where(combined.notna(), MISSING_LABEL).astype(str)
    encoded = pd.get_dummies(categorical, columns=FEATURE_ORDER, dtype=np.float32)
    return encoded.to_numpy(dtype=np.float32), labels


def run_tsne(
    features: np.ndarray,
    perplexity: float,
    max_iter: int,
    random_state: int,
) -> np.ndarray:
    max_perplexity = max(5.0, (len(features) - 1) / 3)
    adjusted_perplexity = min(perplexity, max_perplexity)

    tsne = TSNE(
        n_components=2,
        perplexity=adjusted_perplexity,
        learning_rate="auto",
        init="pca",
        max_iter=max_iter,
        random_state=random_state,
        metric="euclidean",
        n_jobs=1,
    )
    return tsne.fit_transform(features)


def plot_tsne(embedding_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 8), dpi=150)

    real = embedding_df[embedding_df["dataset_source"] == "Real Data"]
    synthetic = embedding_df[embedding_df["dataset_source"] == "Synthetic Data"]

    ax.scatter(
        real["tsne_1"],
        real["tsne_2"],
        s=36,
        c="tab:blue",
        marker="o",
        alpha=0.58,
        edgecolors="white",
        linewidths=0.3,
        label="Real Data",
    )
    ax.scatter(
        synthetic["tsne_1"],
        synthetic["tsne_2"],
        s=34,
        c="tab:orange",
        marker="x",
        alpha=0.78,
        linewidths=1.4,
        label="Synthetic Data",
    )

    ax.set_title("t-SNE Visualization of High-dimensional Manifold Coverage", fontsize=16, pad=14)
    ax.set_xlabel("t-SNE Dimension 1", fontsize=12)
    ax.set_ylabel("t-SNE Dimension 2", fontsize=12)
    ax.grid(True, linestyle="--", linewidth=0.8, alpha=0.35)
    ax.legend(title="Dataset Source", loc="upper right", frameon=True, framealpha=0.95)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a t-SNE plot comparing real and synthetic CHARLS data."
    )
    parser.add_argument("--real-csv", type=Path, default=REAL_DATA_PATH, help="Path to CHARLS_processed_2020.csv.")
    parser.add_argument(
        "--synthetic-csv",
        type=Path,
        default=SYNTHETIC_DATA_PATH,
        help="Path to synthetic_data_v11.csv.",
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="Directory for t-SNE outputs.")
    parser.add_argument("--sample-size", type=int, default=2000, help="Rows sampled from each dataset.")
    parser.add_argument("--perplexity", type=float, default=30.0, help="t-SNE perplexity.")
    parser.add_argument("--max-iter", type=int, default=1000, help="t-SNE maximum iterations.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for sampling and t-SNE.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    real_df = load_real_data(args.real_csv)
    synthetic_df = load_synthetic_data(args.synthetic_csv)
    real_df, synthetic_df = align_columns(real_df, synthetic_df)

    real_sample, synthetic_sample = sample_data(
        real_df,
        synthetic_df,
        sample_size=args.sample_size,
        random_state=args.random_state,
    )
    features, labels = encode_for_tsne(real_sample, synthetic_sample)
    embedding = run_tsne(
        features,
        perplexity=args.perplexity,
        max_iter=args.max_iter,
        random_state=args.random_state,
    )

    embedding_df = pd.DataFrame(
        {
            "tsne_1": embedding[:, 0],
            "tsne_2": embedding[:, 1],
            "dataset_source": labels,
        }
    )
    embedding_df.to_csv(args.output_dir / "tsne_coordinates.csv", index=False)
    plot_tsne(embedding_df, args.output_dir / "tsne_real_vs_synthetic.png")

    print(f"Real rows after fixed preprocessing: {len(real_df)}")
    print(f"Synthetic rows loaded: {len(synthetic_df)}")
    print(f"Sampled real rows: {len(real_sample)}")
    print(f"Sampled synthetic rows: {len(synthetic_sample)}")
    print(f"Encoded feature dimensions: {features.shape[1]}")
    print(f"Saved outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
