from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
from sklearn.metrics import mutual_info_score


REAL_DATA_PATH = Path(r"D:\LLM generate data\data\CHARLS_processed_2020.csv")
MODEL_PATHS = {
    "Ours": Path(r"D:\LLM generate data\data\synthetic_data_v12.csv"),
    "Copula": Path(r"D:\LLM generate data\data\synthetic_data_copula.csv"),
    "TVAE": Path(r"D:\LLM generate data\data\synthetic_data_tvae.csv"),
    "XGB": Path(r"D:\LLM generate data\data\synthetic_data_xgb.csv"),
    "TabPFN": Path(r"D:\LLM generate data\data\synthetic_data_tabpfn.csv"),
    "GReaT": Path(r"D:\LLM generate data\data\synthetic_data_great.csv"),
}
OUTPUT_DIR = Path(__file__).resolve().parent / "multi_model_mi_outputs"

FEATURE_ORDER = [
    "age_bin",
    "gender",
    "marry",
    "edu",
    "income_bin",
    "family_size",
    "health_status",
    "ins",
    "satlife",
    "hospital",
    "social_need",
    "exercise",
]

EPSILON = 1e-12
MODEL_SCORE_COLUMNS = [
    ("mean_gap_score", "Mean"),
    ("std_gap_score", "STD"),
    ("rmse_score", "RMSE"),
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
    df = pd.read_csv(path)
    df = df.dropna()
    return df


def align_columns(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    missing = [col for col in FEATURE_ORDER if col not in df.columns]
    if missing:
        raise ValueError(f"{dataset_name} is missing required columns: {missing}")
    return df[FEATURE_ORDER].copy()


def label_values(series: pd.Series) -> pd.Series:
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
                mi_value = mutual_info_score(label_values(pair[col_i]), label_values(pair[col_j]))

            matrix.loc[col_i, col_j] = mi_value
            matrix.loc[col_j, col_i] = mi_value

    return matrix


def upper_triangle_values(matrix: pd.DataFrame) -> np.ndarray:
    upper_idx = np.triu_indices_from(matrix.to_numpy(), k=1)
    return matrix.to_numpy()[upper_idx]


def median_absolute_deviation(values: np.ndarray) -> float:
    median = np.median(values)
    return float(np.median(np.abs(values - median)))


def safe_pearson_correlation(reference: np.ndarray, candidate: np.ndarray) -> float:
    if np.std(reference) <= EPSILON or np.std(candidate) <= EPSILON:
        return np.nan
    return float(np.corrcoef(reference, candidate)[0, 1])


def cosine_similarity(reference: np.ndarray, candidate: np.ndarray) -> float:
    denominator = float(np.linalg.norm(reference) * np.linalg.norm(candidate))
    if denominator <= EPSILON:
        return np.nan
    return float(np.dot(reference, candidate) / denominator)


def visual_similarity(reference: np.ndarray, candidate: np.ndarray) -> float:
    """Bounded matrix-image similarity based on normalized Frobenius distance."""
    distance = float(np.linalg.norm(candidate - reference))
    scale = float(np.linalg.norm(reference) + np.linalg.norm(candidate))
    if scale <= EPSILON:
        return np.nan
    return float(np.clip(1 - distance / scale, 0, 1))


def histogram_js_divergence(reference: np.ndarray, candidate: np.ndarray, bins: int = 20) -> float:
    max_value = max(float(reference.max()), float(candidate.max()), 0.01)
    bin_edges = np.linspace(0, max_value * 1.05, bins + 1)
    reference_hist, _ = np.histogram(reference, bins=bin_edges)
    candidate_hist, _ = np.histogram(candidate, bins=bin_edges)

    reference_prob = reference_hist.astype(float) + EPSILON
    candidate_prob = candidate_hist.astype(float) + EPSILON
    reference_prob /= reference_prob.sum()
    candidate_prob /= candidate_prob.sum()
    midpoint = 0.5 * (reference_prob + candidate_prob)

    return float(
        0.5 * np.sum(reference_prob * np.log2(reference_prob / midpoint))
        + 0.5 * np.sum(candidate_prob * np.log2(candidate_prob / midpoint))
    )


def safe_filename(name: str) -> str:
    return name.lower().replace(" ", "_").replace("-", "_")


def plot_single_heatmap(
    matrix: pd.DataFrame,
    title: str,
    output_path: Path,
    vmax: float,
) -> None:
    fig, ax = plt.subplots(figsize=(8.6, 7.2))
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
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=45, labelsize=9)
    ax.tick_params(axis="y", rotation=0, labelsize=9)
    plt.setp(ax.get_xticklabels(), ha="right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_heatmap_comparison(
    matrices: dict[str, pd.DataFrame],
    metrics: pd.DataFrame,
    output_path: Path,
    vmax: float,
) -> None:
    fig = plt.figure(figsize=(24, 10.5))
    grid = fig.add_gridspec(
        2,
        4,
        left=0.045,
        right=0.91,
        bottom=0.10,
        top=0.88,
        wspace=0.28,
        hspace=0.36,
    )
    heatmap_slots = [(row, col) for row in range(2) for col in range(4)][:-1]
    if len(matrices) > len(heatmap_slots):
        raise ValueError(f"Comparison figure supports up to {len(heatmap_slots)} heatmaps, got {len(matrices)}.")
    heatmap_positions = heatmap_slots[: len(matrices)]
    heatmap_axes = [fig.add_subplot(grid[row, col]) for row, col in heatmap_positions]
    comparison_ax = fig.add_subplot(grid[1, 3])

    cmap = "YlOrRd"
    norm = Normalize(vmin=0, vmax=vmax)
    for ax, (name, matrix) in zip(heatmap_axes, matrices.items()):
        sns.heatmap(
            matrix,
            ax=ax,
            cmap=cmap,
            vmin=0,
            vmax=vmax,
            square=True,
            cbar=False,
            linewidths=0.15,
            linecolor="white",
        )
        ax.set_title(f"{name} MI Matrix", fontsize=13, fontweight="bold", pad=9)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=45, labelsize=8)
        ax.tick_params(axis="y", rotation=0, labelsize=8)
        plt.setp(ax.get_xticklabels(), ha="right")

    draw_model_comparison_panel(comparison_ax, metrics, cbar=False)
    comparison_ax.set_box_aspect(1)
    comparison_ax.set_anchor("C")

    cbar_ax = fig.add_axes([0.93, 0.17, 0.014, 0.58])
    cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax)
    cbar.set_label("Mutual information", fontsize=10)
    fig.suptitle(
        f"Mutual Information Structure Comparison ({len(matrices) - 1} Models)",
        fontsize=18,
        y=0.97,
    )
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def add_minmax_score(
    metrics: pd.DataFrame,
    source_column: str,
    score_column: str,
    *,
    lower_is_better: bool,
) -> None:
    values = metrics[source_column].astype(float)
    valid_values = values.dropna()
    scores = pd.Series(np.nan, index=metrics.index, dtype=float)

    if valid_values.empty:
        metrics[score_column] = 0.0
        return

    value_min = float(valid_values.min())
    value_max = float(valid_values.max())
    if value_max - value_min <= EPSILON:
        scores.loc[valid_values.index] = 1.0
    elif lower_is_better:
        scores.loc[valid_values.index] = 1 - (valid_values - value_min) / (value_max - value_min)
    else:
        scores.loc[valid_values.index] = (valid_values - value_min) / (value_max - value_min)

    metrics[score_column] = scores.fillna(0.0)


def build_model_metrics(real_matrix: pd.DataFrame, matrices: dict[str, pd.DataFrame]) -> pd.DataFrame:
    real_values = upper_triangle_values(real_matrix)
    real_mean = float(np.mean(real_values))
    real_std = float(np.std(real_values, ddof=0))
    real_mad = median_absolute_deviation(real_values)
    rows = []

    for name, matrix in matrices.items():
        if name == "Real":
            continue

        values = upper_triangle_values(matrix)
        diff = values - real_values
        mean_mi = float(np.mean(values))
        std_mi = float(np.std(values, ddof=0))
        mad_mi = median_absolute_deviation(values)
        rmse = float(np.sqrt(np.mean(diff**2)))
        frobenius_error = float(np.linalg.norm(diff))
        real_norm = float(np.linalg.norm(real_values))
        rows.append(
            {
                "model": name,
                "mean_absolute_error": float(np.mean(np.abs(diff))),
                "root_mean_squared_error": rmse,
                "relative_frobenius_error": frobenius_error / real_norm if real_norm > EPSILON else np.nan,
                "pearson_correlation": safe_pearson_correlation(real_values, values),
                "cosine_similarity": cosine_similarity(real_values, values),
                "visual_similarity": visual_similarity(real_values, values),
                "js_divergence": histogram_js_divergence(real_values, values),
                "mean_mi": mean_mi,
                "real_mean_mi": real_mean,
                "mean_gap": abs(mean_mi - real_mean),
                "std_mi": std_mi,
                "real_std_mi": real_std,
                "std_gap": abs(std_mi - real_std),
                "mad_mi": mad_mi,
                "real_mad_mi": real_mad,
                "mad_gap": abs(mad_mi - real_mad),
            }
        )

    metrics = pd.DataFrame(rows)
    add_minmax_score(metrics, "mean_gap", "mean_gap_score", lower_is_better=True)
    add_minmax_score(metrics, "std_gap", "std_gap_score", lower_is_better=True)
    add_minmax_score(metrics, "mad_gap", "mad_gap_score", lower_is_better=True)
    add_minmax_score(metrics, "root_mean_squared_error", "rmse_score", lower_is_better=True)
    add_minmax_score(metrics, "pearson_correlation", "pearson_score", lower_is_better=False)
    add_minmax_score(metrics, "visual_similarity", "visual_similarity_score", lower_is_better=False)
    metrics["moment_fidelity_score"] = metrics[["mean_gap_score", "std_gap_score"]].mean(axis=1)
    return metrics.sort_values(
        ["moment_fidelity_score", "mean_gap_score", "root_mean_squared_error"],
        ascending=[False, False, True],
    )


def draw_model_comparison_panel(ax: plt.Axes, metrics: pd.DataFrame, *, cbar: bool) -> None:
    score_columns = [column for column, _ in MODEL_SCORE_COLUMNS]
    score_labels = [label for _, label in MODEL_SCORE_COLUMNS]
    score_table = (
        metrics.sort_values("moment_fidelity_score", ascending=False)
        .set_index("model")[score_columns]
        .rename(columns=dict(MODEL_SCORE_COLUMNS))
    )

    sns.heatmap(
        score_table,
        ax=ax,
        cmap="YlGnBu",
        vmin=0,
        vmax=1,
        annot=True,
        fmt=".2f",
        annot_kws={"fontsize": 7},
        linewidths=0.35,
        linecolor="white",
        cbar=cbar,
        cbar_kws=None,
    )
    ax.set_title("Model Fidelity Scores", fontsize=13, fontweight="bold", pad=9)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=0, labelsize=8)
    ax.tick_params(axis="y", rotation=0, labelsize=8)
    ax.set_xticklabels(score_labels)

    if "Ours" in score_table.index:
        row = score_table.index.tolist().index("Ours")
        ax.add_patch(
            Rectangle(
                (0, row),
                score_table.shape[1],
                1,
                fill=False,
                edgecolor="#1B7F3A",
                linewidth=2.0,
                clip_on=False,
            )
        )


def plot_model_comparison(metrics: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.6, 4.6))
    draw_model_comparison_panel(ax, metrics, cbar=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute mutual information heatmaps for real data and multiple synthetic models."
    )
    parser.add_argument("--real-csv", type=Path, default=REAL_DATA_PATH, help="Path to CHARLS_processed_2020.csv.")
    parser.add_argument("--ours-csv", type=Path, default=MODEL_PATHS["Ours"], help="Path to our synthetic CSV.")
    parser.add_argument("--copula-csv", type=Path, default=MODEL_PATHS["Copula"], help="Path to Copula CSV.")
    parser.add_argument("--tvae-csv", type=Path, default=MODEL_PATHS["TVAE"], help="Path to TVAE CSV.")
    parser.add_argument("--xgb-csv", type=Path, default=MODEL_PATHS["XGB"], help="Path to XGB CSV.")
    parser.add_argument("--tabpfn-csv", type=Path, default=MODEL_PATHS["TabPFN"], help="Path to TabPFN CSV.")
    parser.add_argument("--great-csv", type=Path, default=MODEL_PATHS["GReaT"], help="Path to GReaT CSV.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="Directory for figures and CSV files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    datasets = {
        "Real": align_columns(load_real_data(args.real_csv), "Real"),
        "Ours": align_columns(load_synthetic_data(args.ours_csv), "Ours"),
        "Copula": align_columns(load_synthetic_data(args.copula_csv), "Copula"),
        "TVAE": align_columns(load_synthetic_data(args.tvae_csv), "TVAE"),
        "XGB": align_columns(load_synthetic_data(args.xgb_csv), "XGB"),
        "TabPFN": align_columns(load_synthetic_data(args.tabpfn_csv), "TabPFN"),
        "GReaT": align_columns(load_synthetic_data(args.great_csv), "GReaT"),
    }

    matrices = {name: compute_mi_matrix(df, FEATURE_ORDER) for name, df in datasets.items()}
    vmax = max(float(matrix.to_numpy().max()) for matrix in matrices.values())
    vmax = max(vmax, 0.01)

    for name, matrix in matrices.items():
        base = safe_filename(name)
        matrix.to_csv(args.output_dir / f"{base}_mi_matrix.csv")
        plot_single_heatmap(
            matrix,
            f"{name} MI Matrix",
            args.output_dir / f"{base}_mi_heatmap.png",
            vmax=vmax,
        )

    metrics = build_model_metrics(matrices["Real"], matrices)
    metrics.to_csv(args.output_dir / "multi_model_mi_metrics.csv", index=False)
    plot_model_comparison(
        metrics,
        args.output_dir / "multi_model_mi_model_comparison.png",
    )
    plot_heatmap_comparison(
        matrices,
        metrics,
        args.output_dir / "multi_model_mi_heatmap_comparison.png",
        vmax=vmax,
    )

    print("Rows used per dataset:")
    for name, df in datasets.items():
        print(f"  {name}: {len(df)}")
    print(f"Shared heatmap color scale vmax: {vmax:.6f}")
    print(f"Saved outputs to: {args.output_dir}")
    print()
    print("Model MI fidelity metrics:")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
