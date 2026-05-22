from __future__ import annotations

import argparse
import os
import warnings
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
warnings.filterwarnings("ignore", message="Could not find the number of physical cores.*")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import gaussian_kde
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors


DATA_DIR = Path(r"D:\LLM generate data\data")
OUTPUT_DIR = Path(__file__).resolve().parent / "tsne_outputs"
random_state = 42

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

MISSING_LABEL = "__MISSING__"


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
    if insurance_col in df.columns:
        return df

    for fallback_col in ["ins", "pubpen"]:
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
    return df.dropna()


def load_synthetic_data(path: Path, insurance_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = standardize_insurance_column(df, insurance_col)
    return df.dropna()


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


def encode_for_tsne(
    real_sample: pd.DataFrame,
    synthetic_sample: pd.DataFrame,
    columns: list[str],
) -> tuple[np.ndarray, pd.Series]:
    labeled_real = real_sample.copy()
    labeled_real["dataset_source"] = "Real Data"

    labeled_synthetic = synthetic_sample.copy()
    labeled_synthetic["dataset_source"] = "Synthetic Data"

    combined = pd.concat([labeled_real, labeled_synthetic], ignore_index=True)
    labels = combined.pop("dataset_source")

    categorical = combined.astype("object").where(combined.notna(), MISSING_LABEL).astype(str)
    encoded = pd.get_dummies(categorical, columns=columns, dtype=np.float32)
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


def compute_hist_overlap(embedding_df: pd.DataFrame, bins: int = 50) -> float:
    real = embedding_df[embedding_df["dataset_source"] == "Real Data"][["tsne_1", "tsne_2"]].to_numpy()
    synthetic = embedding_df[embedding_df["dataset_source"] == "Synthetic Data"][["tsne_1", "tsne_2"]].to_numpy()
    combined = np.vstack([real, synthetic])

    x_pad = (combined[:, 0].max() - combined[:, 0].min()) * 0.05
    y_pad = (combined[:, 1].max() - combined[:, 1].min()) * 0.05
    ranges = [
        (combined[:, 0].min() - x_pad, combined[:, 0].max() + x_pad),
        (combined[:, 1].min() - y_pad, combined[:, 1].max() + y_pad),
    ]

    real_hist, _, _ = np.histogram2d(real[:, 0], real[:, 1], bins=bins, range=ranges)
    synthetic_hist, _, _ = np.histogram2d(synthetic[:, 0], synthetic[:, 1], bins=bins, range=ranges)
    real_mass = real_hist / max(real_hist.sum(), 1)
    synthetic_mass = synthetic_hist / max(synthetic_hist.sum(), 1)
    return float(np.minimum(real_mass, synthetic_mass).sum())


def compute_mixing_score(embedding_df: pd.DataFrame, n_neighbors: int = 15) -> float:
    coords = embedding_df[["tsne_1", "tsne_2"]].to_numpy()
    labels = embedding_df["dataset_source"].to_numpy()
    k = min(n_neighbors, len(embedding_df) - 1)
    if k <= 0:
        return float("nan")

    neighbors = NearestNeighbors(n_neighbors=k + 1).fit(coords)
    neighbor_indices = neighbors.kneighbors(coords, return_distance=False)[:, 1:]
    opposite_source = labels[neighbor_indices] != labels[:, None]
    return float(opposite_source.mean())


def compute_centroid_delta(embedding_df: pd.DataFrame) -> float:
    real = embedding_df[embedding_df["dataset_source"] == "Real Data"][["tsne_1", "tsne_2"]].to_numpy()
    synthetic = embedding_df[embedding_df["dataset_source"] == "Synthetic Data"][["tsne_1", "tsne_2"]].to_numpy()
    centroid_distance = float(np.linalg.norm(real.mean(axis=0) - synthetic.mean(axis=0)))

    all_coords = embedding_df[["tsne_1", "tsne_2"]].to_numpy()
    pooled_scale = float(np.sqrt(np.mean(np.var(all_coords, axis=0))))
    if pooled_scale == 0:
        return float("nan")
    return centroid_distance / pooled_scale


def compute_metrics(embedding_df: pd.DataFrame) -> dict[str, float]:
    return {
        "overlap": compute_hist_overlap(embedding_df),
        "mixing": compute_mixing_score(embedding_df),
        "centroid_delta": compute_centroid_delta(embedding_df),
    }


def kde_on_grid(points: np.ndarray, grid_x: np.ndarray, grid_y: np.ndarray) -> np.ndarray | None:
    if len(points) < 3:
        return None

    try:
        kde = gaussian_kde(points.T)
        grid_points = np.vstack([grid_x.ravel(), grid_y.ravel()])
        return kde(grid_points).reshape(grid_x.shape)
    except np.linalg.LinAlgError:
        return None


def plot_overlap_panel(
    ax: plt.Axes,
    embedding_df: pd.DataFrame,
    dataset_name: str,
    metrics: dict[str, float],
    compact: bool = False,
) -> None:
    real = embedding_df[embedding_df["dataset_source"] == "Real Data"][["tsne_1", "tsne_2"]].to_numpy()
    synthetic = embedding_df[embedding_df["dataset_source"] == "Synthetic Data"][["tsne_1", "tsne_2"]].to_numpy()
    combined = np.vstack([real, synthetic])

    x_range = combined[:, 0].max() - combined[:, 0].min()
    y_range = combined[:, 1].max() - combined[:, 1].min()
    x_pad = x_range * 0.08
    y_pad = y_range * 0.08
    x_min, x_max = combined[:, 0].min() - x_pad, combined[:, 0].max() + x_pad
    y_min, y_max = combined[:, 1].min() - y_pad, combined[:, 1].max() + y_pad

    xs = np.linspace(x_min, x_max, 140)
    ys = np.linspace(y_min, y_max, 140)
    grid_x, grid_y = np.meshgrid(xs, ys)

    real_density = kde_on_grid(real, grid_x, grid_y)
    synthetic_density = kde_on_grid(synthetic, grid_x, grid_y)

    if real_density is not None:
        ax.contourf(grid_x, grid_y, real_density, levels=9, cmap="Blues", alpha=0.78)
    else:
        ax.scatter(real[:, 0], real[:, 1], s=8, c="tab:blue", alpha=0.18, linewidths=0)

    if synthetic_density is not None:
        levels = np.linspace(float(synthetic_density.min()), float(synthetic_density.max()), 8)[2:]
        ax.contour(grid_x, grid_y, synthetic_density, levels=levels, colors="#E6862A", linewidths=1.5)
    else:
        ax.scatter(synthetic[:, 0], synthetic[:, 1], s=8, c="tab:orange", alpha=0.18, linewidths=0)

    # Light point texture helps reveal sparse synthetic tails without turning the panel into a scatterplot.
    max_points = 450
    if len(synthetic) > max_points:
        rng = np.random.default_rng(7)
        synthetic_plot = synthetic[rng.choice(len(synthetic), max_points, replace=False)]
    else:
        synthetic_plot = synthetic
    ax.scatter(synthetic_plot[:, 0], synthetic_plot[:, 1], s=6, c="#E6862A", alpha=0.16, linewidths=0)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    if compact:
        ax.text(
            0.025,
            0.94,
            dataset_name,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=12,
            fontweight="bold",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 3},
        )
    else:
        ax.set_title(
            (
                f"{dataset_name}\n"
                f"Overlap={metrics['overlap']:.2f}; "
                f"Mixing={metrics['mixing']:.2f}; "
                f"Centroid Δ={metrics['centroid_delta']:.2f}"
            ),
            fontsize=12,
            pad=10,
        )
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("#666666")


def plot_metric_bars(metric_axes: list[plt.Axes], results: list[dict]) -> None:
    dataset_names = [result["name"] for result in results]
    y = np.arange(len(results))
    colors = ["#4C78A8", "#F58518", "#54A24B"]
    metric_specs = [
        ("Overlap", "overlap", "Good ≳ 0.40", max(0.5, max(result["metrics"]["overlap"] for result in results) * 1.15)),
        ("Mixing", "mixing", "Good ≈ 0.40-0.60", max(0.5, max(result["metrics"]["mixing"] for result in results) * 1.15)),
        (
            "Centroid Δ",
            "centroid_delta",
            "Good ≲ 0.20",
            max(0.1, max(result["metrics"]["centroid_delta"] for result in results) * 1.15),
        ),
    ]

    for idx, (ax, (title, key, subtitle, xmax)) in enumerate(zip(metric_axes, metric_specs)):
        values = [result["metrics"][key] for result in results]
        bars = ax.barh(y, values, color=colors, alpha=0.88, height=0.56)
        ax.set_yticks(y, dataset_names)
        ax.invert_yaxis()
        ax.set_xlim(0, xmax)
        ax.set_title(f"{title}\n{subtitle}", fontsize=10.5, pad=8)
        ax.grid(axis="x", linestyle="--", linewidth=0.7, alpha=0.32)
        ax.tick_params(axis="x", labelsize=9)
        ax.tick_params(axis="y", labelsize=10)
        if idx < len(metric_axes) - 1:
            ax.tick_params(axis="x", labelbottom=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)

        if key == "mixing":
            ax.axvline(0.5, color="#555555", linestyle=":", linewidth=1.2)

        for bar, value in zip(bars, values):
            ax.text(
                min(value + xmax * 0.025, xmax * 0.98),
                bar.get_y() + bar.get_height() / 2,
                f"{value:.2f}",
                va="center",
                ha="left" if value < xmax * 0.92 else "right",
                fontsize=9,
                color="#222222",
            )


def plot_three_dataset_overlap(results: list[dict], output_path: Path) -> None:
    fig = plt.figure(figsize=(13.2, 7.4))
    grid = fig.add_gridspec(
        3,
        2,
        width_ratios=[1.15, 0.82],
        left=0.055,
        right=0.985,
        bottom=0.105,
        top=0.88,
        hspace=0.27,
        wspace=0.18,
    )

    contour_axes = [fig.add_subplot(grid[row, 0]) for row in range(len(results))]
    metric_axes = [fig.add_subplot(grid[row, 1]) for row in range(3)]

    for ax, result in zip(contour_axes, results):
        plot_overlap_panel(ax, result["embedding_df"], result["name"], result["metrics"], compact=True)

    plot_metric_bars(metric_axes, results)

    handles = [
        Patch(facecolor="#6BAED6", edgecolor="none", alpha=0.65, label="Real density"),
        Line2D([0], [0], color="#E6862A", linewidth=1.8, label="Synthetic contour"),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#E6862A",
            markersize=5,
            alpha=0.45,
            label="Synthetic sample texture",
        ),
    ]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.055, 0.018), ncol=3, frameon=False, fontsize=10)
    fig.suptitle("t-SNE Manifold Coverage and Summary Metrics", fontsize=15, y=0.96)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def analyze_dataset(config: DatasetConfig, args: argparse.Namespace) -> dict:
    columns = config.feature_order
    real_df = load_real_data(config.real_path, config.insurance_col)
    synthetic_df = load_synthetic_data(config.synthetic_path, config.insurance_col)
    real_df, synthetic_df = align_columns(real_df, synthetic_df, columns, config.name)
    real_sample, synthetic_sample = sample_data(real_df, synthetic_df, args.sample_size, args.random_state)

    features, labels = encode_for_tsne(real_sample, synthetic_sample, columns)
    embedding = run_tsne(features, args.perplexity, args.max_iter, args.random_state)

    embedding_df = pd.DataFrame(
        {
            "tsne_1": embedding[:, 0],
            "tsne_2": embedding[:, 1],
            "dataset_source": labels,
        }
    )
    metrics = compute_metrics(embedding_df)
    embedding_df.to_csv(args.output_dir / f"tsne_coordinates_{config.slug}.csv", index=False)

    return {
        "name": config.name,
        "slug": config.slug,
        "real_rows": len(real_df),
        "synthetic_rows": len(synthetic_df),
        "sampled_real_rows": len(real_sample),
        "sampled_synthetic_rows": len(synthetic_sample),
        "encoded_dimensions": features.shape[1],
        "embedding_df": embedding_df,
        "metrics": metrics,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create t-SNE manifold coverage plots for CHARLS, HRS, and SHARE."
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="Directory for t-SNE outputs.")
    parser.add_argument("--sample-size", type=int, default=2000, help="Rows sampled from each real/synthetic dataset.")
    parser.add_argument("--perplexity", type=float, default=30.0, help="t-SNE perplexity.")
    parser.add_argument("--max-iter", type=int, default=1000, help="t-SNE maximum iterations.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for sampling and t-SNE.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results = [analyze_dataset(config, args) for config in DATASETS]
    plot_three_dataset_overlap(results, args.output_dir / "tsne_three_dataset_overlap.png")

    metrics_rows = []
    for result in results:
        row = {
            "dataset": result["name"],
            "real_rows": result["real_rows"],
            "synthetic_rows": result["synthetic_rows"],
            "sampled_real_rows": result["sampled_real_rows"],
            "sampled_synthetic_rows": result["sampled_synthetic_rows"],
            "encoded_dimensions": result["encoded_dimensions"],
        }
        row.update(result["metrics"])
        metrics_rows.append(row)

    pd.DataFrame(metrics_rows).to_csv(args.output_dir / "tsne_overlap_metrics.csv", index=False)

    for row in metrics_rows:
        print(
            f"{row['dataset']}: real_rows={row['real_rows']:,}, "
            f"synthetic_rows={row['synthetic_rows']:,}, "
            f"sample={row['sampled_real_rows']:,}+{row['sampled_synthetic_rows']:,}, "
            f"overlap={row['overlap']:.3f}, "
            f"mixing={row['mixing']:.3f}, "
            f"centroid_delta={row['centroid_delta']:.3f}"
        )
    print(f"Saved outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
