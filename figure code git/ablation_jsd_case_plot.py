from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon


REAL_DATA_PATH = Path(r"D:\LLM generate data\data\CHARLS_processed_2020.csv")
DATASET_PATHS = {
    "Full Model": Path(r"D:\LLM generate data\data\synthetic_data_v12.csv"),
    "Ablation A": Path(r"D:\LLM generate data\data\synthetic_data_v16.csv"),
    "Ablation B": Path(r"D:\LLM generate data\data\synthetic_data_v17bnew.csv"),
    "Ablation C": Path(r"D:\LLM generate data\data\synthetic_data_v18.csv"),
}
OUTPUT_DIR = Path(__file__).resolve().parent / "jsd_case_outputs"

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

VALUE_ORDER = {
    "age_bin": ["60-", "60-64", "65-69", "70-74", "75-79", "80+"],
    "income_bin": ["Low", "Medium", "High"],
}

OVERALL_DEPENDENCIES = [
    ("edu", "income_bin", "Edu -> Income\n(Social Gradient)"),
    ("ins", "hospital", "Ins -> Hospital\n(Probability)"),
    ("age_bin", "ins", "Age -> Ins\n(Hard Constraint)"),
    ("exercise", "social_need", "Exercise -> Social\n(Subjective)"),
    ("family_size", "satlife", "Familysize -> Satlife\n(Long-tail)"),
]

SPECIFIC_CASES = [
    ("satlife", "hospital", "1.0", "Satlife -> Hospital\n(Group 1.0)"),
    ("family_size", "ins", "6.0", "Familysize -> Ins\n(Group 6.0)"),
    ("age_bin", "social_need", "60-64", "Age -> Social\n(Group 60-64)"),
]


def load_real_data(path: Path) -> pd.DataFrame:
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
    #df = df.dropna()
    return df


def align_columns(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    missing = [col for col in FEATURE_ORDER if col not in df.columns]
    if missing:
        raise ValueError(f"{dataset_name} is missing required columns: {missing}")
    return df[FEATURE_ORDER].copy()


def canonical_label(value: object) -> str:
    if pd.isna(value):
        return "__MISSING__"
    raw = str(value).strip()
    try:
        numeric = float(raw)
        if numeric.is_integer():
            return f"{numeric:.1f}"
        return f"{numeric:g}"
    except ValueError:
        return raw


def label_series(series: pd.Series) -> pd.Series:
    return series.map(canonical_label).astype(str)


def value_sort_key(column: str, value: str) -> tuple[int, float | str]:
    ordered = VALUE_ORDER.get(column)
    if ordered and value in ordered:
        return (0, ordered.index(value))
    try:
        return (1, float(value))
    except ValueError:
        return (2, value)


def sorted_categories(real: pd.Series, synthetic: pd.Series, column: str) -> list[str]:
    labels = set(label_series(real)).union(set(label_series(synthetic)))
    return sorted(labels, key=lambda value: value_sort_key(column, value))


def distribution_jsd(real_values: pd.Series, synthetic_values: pd.Series, categories: list[str]) -> float:
    if len(real_values) == 0 or len(synthetic_values) == 0:
        return 1.0
    real_dist = real_values.value_counts(normalize=True).reindex(categories, fill_value=0.0)
    synthetic_dist = synthetic_values.value_counts(normalize=True).reindex(categories, fill_value=0.0)
    return float(jensenshannon(real_dist.to_numpy(), synthetic_dist.to_numpy(), base=2.0))


def conditional_jsd_values(
    real_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    condition_col: str,
    target_col: str,
    min_group_size: int,
) -> dict[str, float]:
    real_condition = label_series(real_df[condition_col])
    synthetic_condition = label_series(synthetic_df[condition_col])
    real_target = label_series(real_df[target_col])
    synthetic_target = label_series(synthetic_df[target_col])

    condition_values = sorted(
        set(real_condition).union(set(synthetic_condition)),
        key=lambda value: value_sort_key(condition_col, value),
    )
    target_values = sorted_categories(real_df[target_col], synthetic_df[target_col], target_col)

    jsd_by_group = {}
    for condition_value in condition_values:
        real_values = real_target.loc[real_condition == condition_value]
        synthetic_values = synthetic_target.loc[synthetic_condition == condition_value]
        if len(real_values) < min_group_size or len(synthetic_values) < min_group_size:
            jsd_by_group[condition_value] = 1.0
        else:
            jsd_by_group[condition_value] = distribution_jsd(real_values, synthetic_values, target_values)
    return jsd_by_group


def build_overall_jsd_matrix(
    real_df: pd.DataFrame,
    datasets: dict[str, pd.DataFrame],
    min_group_size: int,
) -> pd.DataFrame:
    rows = []
    for condition_col, target_col, row_label in OVERALL_DEPENDENCIES:
        row = {"Dependency": row_label}
        for dataset_name, synthetic_df in datasets.items():
            jsd_by_group = conditional_jsd_values(
                real_df=real_df,
                synthetic_df=synthetic_df,
                condition_col=condition_col,
                target_col=target_col,
                min_group_size=min_group_size,
            )
            row[dataset_name] = float(np.mean(list(jsd_by_group.values()))) if jsd_by_group else np.nan
        rows.append(row)
    return pd.DataFrame(rows).set_index("Dependency")


def build_specific_case_table(
    real_df: pd.DataFrame,
    datasets: dict[str, pd.DataFrame],
    min_group_size: int,
) -> pd.DataFrame:
    rows = []
    for condition_col, target_col, condition_value, case_label in SPECIFIC_CASES:
        row = {"Case": case_label}
        condition_value = canonical_label(condition_value)
        for dataset_name, synthetic_df in datasets.items():
            jsd_by_group = conditional_jsd_values(
                real_df=real_df,
                synthetic_df=synthetic_df,
                condition_col=condition_col,
                target_col=target_col,
                min_group_size=min_group_size,
            )
            row[dataset_name] = jsd_by_group.get(condition_value, np.nan)
        rows.append(row)
    return pd.DataFrame(rows).set_index("Case")


def draw_figure(overall_df: pd.DataFrame, case_df: pd.DataFrame, output_path: Path) -> None:
    fig = plt.figure(figsize=(13.6, 4.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.12, 1.0], wspace=0.24)

    ax_heatmap = fig.add_subplot(gs[0, 0])
    matrix = overall_df.to_numpy(dtype=float)
    vmin = 0.0
    vmax = max(0.30, float(np.nanmax(matrix)))
    image = ax_heatmap.imshow(matrix, cmap="YlOrRd", vmin=vmin, vmax=vmax, aspect="auto")

    ax_heatmap.set_title("(a) Overall Distributional Consistency (Avg. JSD)", fontsize=9, fontweight="bold", pad=10)
    ax_heatmap.set_xticks(np.arange(overall_df.shape[1]))
    column_labels = {
        "Full Model": "Full Model",
        "Ablation A": "Ablation A\n(w/o Semantic)",
        "Ablation B": "Ablation B\n(w/o Proposal)",
        "Ablation C": "Ablation C\n(w/o Dual)"
    }
    ax_heatmap.set_xticklabels([column_labels.get(col, col) for col in overall_df.columns], fontsize=8, fontweight="bold")
    ax_heatmap.set_yticks(np.arange(overall_df.shape[0]))
    ax_heatmap.set_yticklabels(overall_df.index, fontsize=8)

    for row_idx in range(overall_df.shape[0]):
        for col_idx in range(overall_df.shape[1]):
            value = matrix[row_idx, col_idx]
            text_color = "white" if value >= vmax * 0.58 else "black"
            ax_heatmap.text(col_idx, row_idx, f"{value:.3f}", ha="center", va="center", color=text_color, fontsize=8)

    ax_heatmap.set_xticks(np.arange(-0.5, overall_df.shape[1], 1), minor=True)
    ax_heatmap.set_yticks(np.arange(-0.5, overall_df.shape[0], 1), minor=True)
    ax_heatmap.grid(which="minor", color="#777777", linewidth=0.7)
    ax_heatmap.tick_params(which="minor", bottom=False, left=False)
    colorbar = fig.colorbar(image, ax=ax_heatmap, fraction=0.046, pad=0.055)
    colorbar.set_label("Avg. JSD (Lower is Better)", fontsize=8)
    colorbar.ax.tick_params(labelsize=8)

    ax_bar = fig.add_subplot(gs[0, 1])
    x = np.arange(case_df.shape[0])
    width = min(0.2, 0.8 / max(case_df.shape[1], 1))
    colors = {
        "Full Model": "#ff7f0e",
        "Ablation A": "#2ca02c",
        "Ablation B": "#d62728",
        "Ablation C": "#9467bd",
    }
    for idx, dataset_name in enumerate(case_df.columns):
        offset = (idx - (case_df.shape[1] - 1) / 2) * width
        ax_bar.bar(
            x + offset,
            case_df[dataset_name].to_numpy(dtype=float),
            width=width,
            color=colors.get(dataset_name, "#7f7f7f"),
            edgecolor="#555555",
            alpha=0.45,
            label=dataset_name,
        )

    ax_bar.set_title("(b) Specific Case Analysis: Structural Collapse", fontsize=9, fontweight="bold", pad=10)
    ax_bar.set_ylabel("JSD (Specific Group)", fontsize=8)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(case_df.index, fontsize=8)
    ax_bar.tick_params(axis="y", labelsize=8)
    ax_bar.grid(axis="y", linestyle="--", alpha=0.45)
    ax_bar.legend(loc="upper left", fontsize=8, frameon=True)
    ax_bar.set_ylim(0, max(0.72, float(np.nanmax(case_df.to_numpy(dtype=float))) * 1.18))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def print_table(title: str, table: pd.DataFrame) -> None:
    print()
    print(title.center(120))
    print()
    formatted = table.copy()
    for col in formatted.columns:
        formatted[col] = formatted[col].map(lambda value: f"{value:.3f}" if pd.notna(value) else "NA")
    print(formatted.to_string())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot ablation JSD heatmap and specific-case bar chart.")
    parser.add_argument("--real-csv", type=Path, default=REAL_DATA_PATH, help="Path to CHARLS_processed_2020.csv.")
    parser.add_argument("--full-csv", type=Path, default=DATASET_PATHS["Full Model"], help="Path to full-model CSV.")
    parser.add_argument("--ablation-a-csv", type=Path, default=DATASET_PATHS["Ablation A"], help="Path to ablation A CSV.")
    parser.add_argument("--ablation-b-csv", type=Path, default=DATASET_PATHS["Ablation B"], help="Path to ablation B CSV.")
    parser.add_argument("--ablation-c-csv", type=Path, default=DATASET_PATHS["Ablation C"], help="Path to ablation C CSV.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="Directory for JSD case outputs.")
    parser.add_argument("--min-group-size", type=int, default=1, help="Minimum rows per conditional group.")
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

    overall_df = build_overall_jsd_matrix(real_df, datasets, min_group_size=args.min_group_size)
    case_df = build_specific_case_table(real_df, datasets, min_group_size=args.min_group_size)

    output_path = args.output_dir / "ablation_jsd_case_analysis.png"
    draw_figure(overall_df, case_df, output_path)

    pd.set_option("display.width", 180)
    print_table("OVERALL DISTRIBUTIONAL CONSISTENCY (AVG. JSD)", overall_df)
    print_table("SPECIFIC CASE ANALYSIS (JSD)", case_df)
    print(f"\nSaved figure to: {output_path}")


if __name__ == "__main__":
    main()
