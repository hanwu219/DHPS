from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mutual_info_score


REAL_DATA_PATH = Path(r"D:\LLM generate data\data\CHARLS_processed_2020.csv")
SYNTHETIC_DATA_PATH = Path(r"D:\LLM generate data\data\synthetic_data_v11.csv")
OUTPUT_DIR = Path(__file__).resolve().parent / "mi_outputs"

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
    """Load synthetic data. Missing values are handled pairwise during MI."""
    df = pd.read_csv(path)
    df = df.dropna()
    return df


def align_columns(real_df: pd.DataFrame, synthetic_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    missing_real = [col for col in FEATURE_ORDER if col not in real_df.columns]
    missing_synthetic = [col for col in FEATURE_ORDER if col not in synthetic_df.columns]
    if missing_real or missing_synthetic:
        raise ValueError(
            "Column mismatch. "
            f"Missing in real data: {missing_real}; "
            f"missing in synthetic data: {missing_synthetic}"
        )

    return real_df[FEATURE_ORDER].copy(), synthetic_df[FEATURE_ORDER].copy(), FEATURE_ORDER.copy()


def _label_values(series: pd.Series) -> pd.Series:
    """Convert values to stable discrete labels for sklearn mutual_info_score."""
    return series.astype("object").astype(str)


def compute_mi_matrix(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    matrix = pd.DataFrame(0.0, index=columns, columns=columns)

    for i, col_i in enumerate(columns):
        for j, col_j in enumerate(columns):
            if i >= j:
                continue

            pair = df[[col_i, col_j]].dropna()
            if pair.empty:
                mi_value = 0.0
            else:
                mi_value = mutual_info_score(
                    _label_values(pair[col_i]),
                    _label_values(pair[col_j]),
                )

            matrix.loc[col_i, col_j] = mi_value
            matrix.loc[col_j, col_i] = mi_value

    return matrix


def upper_triangle_values(matrix: pd.DataFrame) -> np.ndarray:
    upper_idx = np.triu_indices_from(matrix.to_numpy(), k=1)
    return matrix.to_numpy()[upper_idx]


def plot_heatmap(
    matrix: pd.DataFrame,
    title: str,
    output_path: Path,
    vmax: float | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        matrix,
        ax=ax,
        cmap="YlOrRd",
        vmin=0,
        vmax=vmax,
        square=True,
        cbar=True,
        linewidths=0.2,
        linecolor="white",
    )
    ax.set_title(title, fontsize=14, pad=12)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="y", rotation=0)
    plt.setp(ax.get_xticklabels(), ha="right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_distribution_comparison(
    real_values: np.ndarray,
    synthetic_values: np.ndarray,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    max_value = max(float(real_values.max()), float(synthetic_values.max()), 0.01)
    bins = np.linspace(0, max_value * 1.05, 30)

    real_mean = float(real_values.mean())
    synthetic_mean = float(synthetic_values.mean())
    real_std = float(real_values.std(ddof=0))
    synthetic_std = float(synthetic_values.std(ddof=0))

    ax.hist(real_values, bins=bins, density=True, alpha=0.45, label="Real Data Dist.")
    ax.hist(synthetic_values, bins=bins, density=True, alpha=0.45, label="Synthetic Data Dist.")

    ax.axvline(real_mean, color="tab:blue", linewidth=2, label=f"Real Data Mean: {real_mean:.3f}")
    ax.axvline(
        synthetic_mean,
        color="tab:orange",
        linewidth=2,
        label=f"Synthetic Data Mean: {synthetic_mean:.3f}",
    )
    ax.axvline(real_mean - real_std, color="tab:blue", linestyle=":", linewidth=1.5)
    ax.axvline(
        real_mean + real_std,
        color="tab:blue",
        linestyle=":",
        linewidth=1.5,
        label=f"Real Data Std: {real_std:.3f}",
    )
    ax.axvline(synthetic_mean - synthetic_std, color="tab:orange", linestyle=":", linewidth=1.5)
    ax.axvline(
        synthetic_mean + synthetic_std,
        color="tab:orange",
        linestyle=":",
        linewidth=1.5,
        label=f"Synthetic Data Std: {synthetic_std:.3f}",
    )

    ax.set_title("Global Distribution of Mutual Information", fontsize=14)
    ax.set_xlabel("Mutual Information Value")
    ax.set_ylabel("Density")
    ax.legend(loc="upper right", frameon=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_overview(
    real_matrix: pd.DataFrame,
    synthetic_matrix: pd.DataFrame,
    real_values: np.ndarray,
    synthetic_values: np.ndarray,
    output_path: Path,
) -> None:
    vmax = max(float(real_matrix.to_numpy().max()), float(synthetic_matrix.to_numpy().max()), 0.01)

    fig = plt.figure(figsize=(20, 7))
    grid = fig.add_gridspec(
        1,
        3,
        width_ratios=[1.05, 1.05, 1.25],
        left=0.06,
        right=0.98,
        bottom=0.24,
        top=0.80,
        wspace=0.36,
    )
    axes = [fig.add_subplot(grid[0, idx]) for idx in range(3)]

    for ax, matrix, title in [
        (axes[0], real_matrix, "Real Data MI Matrix"),
        (axes[1], synthetic_matrix, "Synthetic Data MI Matrix"),
    ]:
        sns.heatmap(
            matrix,
            ax=ax,
            cmap="YlOrRd",
            vmin=0,
            vmax=vmax,
            square=False,
            cbar=True,
            cbar_kws={"shrink": 0.95},
            linewidths=0.2,
            linecolor="white",
        )
        ax.set_title(title, fontsize=14, pad=12)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=45)
        ax.tick_params(axis="y", rotation=0)
        plt.setp(ax.get_xticklabels(), ha="right")

    real_mean = float(real_values.mean())
    synthetic_mean = float(synthetic_values.mean())
    real_std = float(real_values.std(ddof=0))
    synthetic_std = float(synthetic_values.std(ddof=0))
    max_value = max(float(real_values.max()), float(synthetic_values.max()), 0.01)
    bins = np.linspace(0, max_value * 1.05, 30)

    axes[2].hist(real_values, bins=bins, density=True, alpha=0.45, label="Real Data Dist.")
    axes[2].hist(synthetic_values, bins=bins, density=True, alpha=0.45, label="Synthetic Data Dist.")
    axes[2].axvline(real_mean, color="tab:blue", linewidth=2, label=f"Real Data Mean: {real_mean:.3f}")
    axes[2].axvline(
        synthetic_mean,
        color="tab:orange",
        linewidth=2,
        label=f"Synthetic Data Mean: {synthetic_mean:.3f}",
    )
    axes[2].axvline(real_mean - real_std, color="tab:blue", linestyle=":", linewidth=1.5)
    axes[2].axvline(
        real_mean + real_std,
        color="tab:blue",
        linestyle=":",
        linewidth=1.5,
        label=f"Real Data Std: {real_std:.3f}",
    )
    axes[2].axvline(synthetic_mean - synthetic_std, color="tab:orange", linestyle=":", linewidth=1.5)
    axes[2].axvline(
        synthetic_mean + synthetic_std,
        color="tab:orange",
        linestyle=":",
        linewidth=1.5,
        label=f"Synthetic Data Std: {synthetic_std:.3f}",
    )
    axes[2].set_title("Global Distribution of Mutual Information", fontsize=14)
    axes[2].set_xlabel("Mutual Information Value")
    axes[2].set_ylabel("Density")
    axes[2].legend(loc="upper right", frameon=True)

    fig.suptitle("Mutual Information", fontsize=18, y=0.96)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_pairwise_comparison(
    real_matrix: pd.DataFrame,
    synthetic_matrix: pd.DataFrame,
    output_path: Path,
) -> None:
    rows = []
    columns = real_matrix.columns.tolist()
    for i, col_i in enumerate(columns):
        for j in range(i + 1, len(columns)):
            col_j = columns[j]
            real_mi = float(real_matrix.loc[col_i, col_j])
            synthetic_mi = float(synthetic_matrix.loc[col_i, col_j])
            rows.append(
                {
                    "variable_1": col_i,
                    "variable_2": col_j,
                    "real_mi": real_mi,
                    "synthetic_mi": synthetic_mi,
                    "absolute_diff": abs(real_mi - synthetic_mi),
                }
            )

    pd.DataFrame(rows).sort_values("absolute_diff", ascending=False).to_csv(
        output_path,
        index=False,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute and plot mutual information matrices for real and synthetic CHARLS data."
    )
    parser.add_argument("--real-csv", type=Path, default=REAL_DATA_PATH, help="Path to CHARLS_processed_2020.csv.")
    parser.add_argument(
        "--synthetic-csv",
        type=Path,
        default=SYNTHETIC_DATA_PATH,
        help="Path to synthetic_data_v11.csv.",
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="Directory for figures and CSV files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    real_df = load_real_data(args.real_csv)
    synthetic_df = load_synthetic_data(args.synthetic_csv)
    real_df, synthetic_df, columns = align_columns(real_df, synthetic_df)

    real_matrix = compute_mi_matrix(real_df, columns)
    synthetic_matrix = compute_mi_matrix(synthetic_df, columns)
    real_values = upper_triangle_values(real_matrix)
    synthetic_values = upper_triangle_values(synthetic_matrix)

    vmax = max(float(real_matrix.to_numpy().max()), float(synthetic_matrix.to_numpy().max()), 0.01)
    plot_heatmap(real_matrix, "Real Data MI Matrix", args.output_dir / "real_mi_heatmap.png", vmax=vmax)
    plot_heatmap(
        synthetic_matrix,
        "Synthetic Data MI Matrix",
        args.output_dir / "synthetic_mi_heatmap.png",
        vmax=vmax,
    )
    plot_distribution_comparison(
        real_values,
        synthetic_values,
        args.output_dir / "mi_distribution_comparison.png",
    )
    plot_overview(
        real_matrix,
        synthetic_matrix,
        real_values,
        synthetic_values,
        args.output_dir / "mi_overview.png",
    )

    real_matrix.to_csv(args.output_dir / "real_mi_matrix.csv")
    synthetic_matrix.to_csv(args.output_dir / "synthetic_mi_matrix.csv")
    save_pairwise_comparison(
        real_matrix,
        synthetic_matrix,
        args.output_dir / "mi_pairwise_comparison.csv",
    )

    print(f"Real rows after fixed preprocessing: {len(real_df)}")
    print(f"Synthetic rows loaded: {len(synthetic_df)}")
    print(f"Saved outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
