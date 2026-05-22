from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mutual_info_score


DATA_DIR = Path(r"D:\LLM generate data\data")
OUTPUT_DIR = Path(__file__).resolve().parent / "mi_outputs"
HEATMAP_VMIN = 0.0
HEATMAP_VMAX = 0.2

BASE_FEATURE_ORDER = [
    "age_bin",
    "gender",
    "marry",
    "edu",
    "income_bin",
    "family_size",
    "health_status",
    "hospital",
    "exercise",
    "{insurance_col}",
    "satlife",
    "social_need",
]


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    real_path: Path
    synthetic_path: Path
    insurance_col: str = "ins"

    @property
    def slug(self) -> str:
        return self.name.lower()

    @property
    def feature_order(self) -> list[str]:
        return [col.format(insurance_col=self.insurance_col) for col in BASE_FEATURE_ORDER]


DATASETS = [
    DatasetConfig(
        name="CHARLS",
        real_path=DATA_DIR / "CHARLS_processed_2020.csv",
        synthetic_path=DATA_DIR / "synthetic_data_v11.csv",
    ),
    DatasetConfig(
        name="HRS",
        real_path=DATA_DIR / "HRS_processed_2020.csv",
        synthetic_path=DATA_DIR / "synthetic_data_hrs.csv",
    ),
    DatasetConfig(
        name="SHARE",
        real_path=DATA_DIR / "SHARE_processed_2020.csv",
        synthetic_path=DATA_DIR / "synthetic_data_share.csv",
        insurance_col="pubpen",
    ),
]


def standardize_insurance_column(df: pd.DataFrame, insurance_col: str) -> pd.DataFrame:
    """Use the dataset-specific insurance/proxy column name in downstream plots."""
    if insurance_col in df.columns:
        return df

    fallback_cols = ["ins", "pubpen"]
    for fallback_col in fallback_cols:
        if fallback_col in df.columns:
            return df.rename(columns={fallback_col: insurance_col})

    return df


def load_real_data(path: Path, insurance_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = standardize_insurance_column(df, insurance_col)

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


def load_synthetic_data(path: Path, insurance_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = standardize_insurance_column(df, insurance_col)
    return df


def align_columns(
    real_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    columns: list[str],
    dataset_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    missing_real = [col for col in columns if col not in real_df.columns]
    missing_synthetic = [col for col in columns if col not in synthetic_df.columns]
    if missing_real or missing_synthetic:
        raise ValueError(
            f"{dataset_name} column mismatch. "
            f"Missing in real data: {missing_real}; "
            f"missing in synthetic data: {missing_synthetic}"
        )

    return real_df[columns].copy(), synthetic_df[columns].copy()


def _label_values(series: pd.Series) -> pd.Series:
    return series.astype("object").astype(str)


def compute_mi_matrix(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    matrix = pd.DataFrame(0.0, index=columns, columns=columns)

    for i, col_i in enumerate(columns):
        for j in range(i + 1, len(columns)):
            col_j = columns[j]
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


def summarize_difference(real_matrix: pd.DataFrame, synthetic_matrix: pd.DataFrame) -> dict[str, float]:
    signed_values = upper_triangle_values(synthetic_matrix - real_matrix)
    abs_values = np.abs(signed_values)
    real_values = upper_triangle_values(real_matrix)
    synthetic_values = upper_triangle_values(synthetic_matrix)

    if np.std(real_values) == 0 or np.std(synthetic_values) == 0:
        pearson_r = np.nan
    else:
        pearson_r = float(np.corrcoef(real_values, synthetic_values)[0, 1])

    return {
        "mean_abs_diff": float(abs_values.mean()),
        "std_abs_diff": float(abs_values.std(ddof=0)),
        "rmse": float(np.sqrt(np.mean(signed_values**2))),
        "max_abs_diff": float(abs_values.max()),
        "signed_mean_diff": float(signed_values.mean()),
        "pearson_r": pearson_r,
    }


def plot_single_difference_heatmap(
    matrix: pd.DataFrame,
    dataset_name: str,
    metrics: dict[str, float],
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8.6, 7.4))
    sns.heatmap(
        matrix,
        ax=ax,
        cmap="YlOrRd",
        vmin=HEATMAP_VMIN,
        vmax=HEATMAP_VMAX,
        square=True,
        cbar=True,
        cbar_kws={"label": "|Synthetic MI - Real MI|"},
        linewidths=0.2,
        linecolor="white",
    )
    ax.set_title(
        (
            f"{dataset_name}: Absolute MI Difference\n"
            f"Mean |Δ|={metrics['mean_abs_diff']:.3f}   "
            f"Std |Δ|={metrics['std_abs_diff']:.3f}   "
            f"RMSE={metrics['rmse']:.3f}"
        ),
        fontsize=13,
        pad=12,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="y", rotation=0)
    plt.setp(ax.get_xticklabels(), ha="right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_three_dataset_overview(
    results: list[dict],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(
        1,
        len(results),
        figsize=(18, 6.6),
        gridspec_kw={"left": 0.05, "right": 0.91, "bottom": 0.21, "top": 0.78, "wspace": 0.28},
    )
    if len(results) == 1:
        axes = [axes]

    last_heatmap = None
    for ax, result in zip(axes, results):
        metrics = result["metrics"]
        last_heatmap = sns.heatmap(
            result["abs_diff_matrix"],
            ax=ax,
            cmap="YlOrRd",
            vmin=HEATMAP_VMIN,
            vmax=HEATMAP_VMAX,
            square=True,
            cbar=False,
            linewidths=0.2,
            linecolor="white",
        )
        ax.set_title(
            (
                f"{result['name']}\n"
                f"Mean |Δ|={metrics['mean_abs_diff']:.3f}; "
                f"Std |Δ|={metrics['std_abs_diff']:.3f}; "
                f"RMSE={metrics['rmse']:.3f}"
            ),
            fontsize=12,
            pad=10,
        )
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=45, labelsize=9)
        ax.tick_params(axis="y", rotation=0, labelsize=9)
        plt.setp(ax.get_xticklabels(), ha="right")

    if last_heatmap is not None:
        cbar_ax = fig.add_axes([0.925, 0.23, 0.015, 0.52])
        fig.colorbar(last_heatmap.collections[0], cax=cbar_ax, label="|Synthetic MI - Real MI|")

    fig.suptitle("Mutual Information Difference Between Real and Synthetic Data", fontsize=16, y=0.94)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_pairwise_comparison(
    dataset_name: str,
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
            signed_diff = synthetic_mi - real_mi
            rows.append(
                {
                    "dataset": dataset_name,
                    "variable_1": col_i,
                    "variable_2": col_j,
                    "real_mi": real_mi,
                    "synthetic_mi": synthetic_mi,
                    "signed_diff": signed_diff,
                    "absolute_diff": abs(signed_diff),
                }
            )

    pd.DataFrame(rows).sort_values("absolute_diff", ascending=False).to_csv(output_path, index=False)


def analyze_dataset(config: DatasetConfig, output_dir: Path) -> dict:
    columns = config.feature_order
    real_df = load_real_data(config.real_path, config.insurance_col)
    synthetic_df = load_synthetic_data(config.synthetic_path, config.insurance_col)
    real_df, synthetic_df = align_columns(real_df, synthetic_df, columns, config.name)

    real_matrix = compute_mi_matrix(real_df, columns)
    synthetic_matrix = compute_mi_matrix(synthetic_df, columns)
    abs_diff_matrix = (synthetic_matrix - real_matrix).abs()
    metrics = summarize_difference(real_matrix, synthetic_matrix)

    real_matrix.to_csv(output_dir / f"{config.slug}_real_mi_matrix.csv")
    synthetic_matrix.to_csv(output_dir / f"{config.slug}_synthetic_mi_matrix.csv")
    abs_diff_matrix.to_csv(output_dir / f"{config.slug}_abs_mi_diff_matrix.csv")
    save_pairwise_comparison(
        config.name,
        real_matrix,
        synthetic_matrix,
        output_dir / f"{config.slug}_mi_pairwise_comparison.csv",
    )

    return {
        "name": config.name,
        "slug": config.slug,
        "real_rows": len(real_df),
        "synthetic_rows": len(synthetic_df),
        "real_matrix": real_matrix,
        "synthetic_matrix": synthetic_matrix,
        "abs_diff_matrix": abs_diff_matrix,
        "metrics": metrics,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot three dataset-level mutual-information difference heatmaps."
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="Directory for figures and CSV files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results = [analyze_dataset(config, args.output_dir) for config in DATASETS]

    for result in results:
        plot_single_difference_heatmap(
            result["abs_diff_matrix"],
            result["name"],
            result["metrics"],
            args.output_dir / f"{result['slug']}_abs_mi_diff_heatmap.png",
        )

    plot_three_dataset_overview(results, args.output_dir / "mi_abs_diff_three_datasets.png")

    summary_rows = []
    for result in results:
        row = {
            "dataset": result["name"],
            "real_rows": result["real_rows"],
            "synthetic_rows": result["synthetic_rows"],
        }
        row.update(result["metrics"])
        summary_rows.append(row)

    pd.DataFrame(summary_rows).to_csv(args.output_dir / "mi_summary_metrics.csv", index=False)

    for row in summary_rows:
        print(
            f"{row['dataset']}: real_rows={row['real_rows']:,}, "
            f"synthetic_rows={row['synthetic_rows']:,}, "
            f"mean_abs_diff={row['mean_abs_diff']:.4f}, "
            f"std_abs_diff={row['std_abs_diff']:.4f}, "
            f"rmse={row['rmse']:.4f}"
        )
    print(f"Saved outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
