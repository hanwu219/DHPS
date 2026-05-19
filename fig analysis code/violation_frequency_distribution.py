from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
from scipy.stats import gaussian_kde


REAL_DATA_PATH = Path(r"D:\LLM generate data\data\CHARLS_processed_2020.csv")
DATASET_PATHS = {
    "Full Model": Path(r"D:\LLM generate data\data\synthetic_data_v12.csv"),
    "Ablation A": Path(r"D:\LLM generate data\data\synthetic_data_v16.csv"),
    "Ablation B": Path(r"D:\LLM generate data\data\synthetic_data_v17bnew.csv"),
    "Ablation C": Path(r"D:\LLM generate data\data\synthetic_data_v18.csv"),
}
OUTPUT_DIR = Path(__file__).resolve().parent / "violation_outputs"

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

H_COLS = ["hospital", "exercise", "ins", "satlife", "social_need"]

# Default validated edges are the graph bootstrap joints shown in graph_v2_charls_ipf.py output.
DEFAULT_BOOTSTRAP_EDGES = [
    ("age_bin", "hospital"),
    ("age_bin", "exercise"),
    ("age_bin", "ins"),
    ("age_bin", "satlife"),
    ("age_bin", "social_need"),
    ("income_bin", "social_need"),
    ("family_size", "satlife"),
    ("family_size", "social_need"),
    ("ins", "marry"),
    ("marry", "satlife"),
    ("edu", "satlife"),
    ("edu", "social_need"),
    ("health_status", "satlife"),
    ("exercise", "satlife"),
    ("exercise", "social_need"),
    ("ins", "satlife"),
    ("ins", "social_need"),
    ("satlife", "social_need"),
]

MISSING_LABEL = "__MISSING__"
DEFAULT_NEAR_ZERO_THRESHOLD = 0.005


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
    #df = df.dropna()
    return df


def align_columns(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    missing = [col for col in FEATURE_ORDER if col not in df.columns]
    if missing:
        raise ValueError(f"{dataset_name} is missing required columns: {missing}")
    return df[FEATURE_ORDER].copy()


def label_series(series: pd.Series) -> pd.Series:
    return series.astype("object").where(series.notna(), MISSING_LABEL).astype(str)


def value_mask(series: pd.Series, target_label: str) -> pd.Series:
    labels = label_series(series)
    target_numeric = pd.to_numeric(pd.Series([target_label]), errors="coerce").iloc[0]
    if pd.notna(target_numeric):
        numeric_series = pd.to_numeric(series, errors="coerce")
        return np.isclose(numeric_series.astype(float), float(target_numeric), equal_nan=False)
    return labels == target_label


def value_sort_key(value: str) -> tuple[int, float | str]:
    if value == MISSING_LABEL:
        return (2, value)
    try:
        return (0, float(value))
    except ValueError:
        return (1, value)


def bootstrap_categories(series: pd.Series) -> list[str]:
    if isinstance(series.dtype, CategoricalDtype):
        return [str(value) for value in series.cat.categories]
    return sorted(label_series(series).unique().tolist(), key=value_sort_key)


def load_edges(edges_json: Path | None) -> list[tuple[str, str]]:
    if edges_json is None:
        return DEFAULT_BOOTSTRAP_EDGES.copy()

    with open(edges_json, "r", encoding="utf-8") as file:
        raw_edges = json.load(file)

    edges = []
    for edge in raw_edges:
        if isinstance(edge, dict):
            edge = edge.get("pair", [])
        if len(edge) != 2:
            raise ValueError(f"Invalid edge entry: {edge}")
        edges.append((str(edge[0]), str(edge[1])))
    return edges


def validate_edges(edges: Iterable[tuple[str, str]]) -> list[tuple[str, str]]:
    validated = []
    for col_a, col_b in edges:
        if col_a not in FEATURE_ORDER or col_b not in FEATURE_ORDER:
            raise ValueError(f"Unknown edge columns: {col_a}, {col_b}")
        validated.append((col_a, col_b))
    return validated


def load_bootstrap_target(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)
    if "s_target" in data:
        data = data["s_target"]
    if "joints" not in data:
        raise ValueError("Bootstrap target JSON must contain either 'joints' or 's_target.joints'.")
    return data


def build_bootstrap_target(
    real_sample: pd.DataFrame,
    h_cols: list[str],
    edges: list[tuple[str, str]],
    n_iterations: int,
    bootstrap_random_state: int | None,
) -> dict:
    marginal_acc = {col: {} for col in h_cols}
    joint_acc = {f"{col_a}|{col_b}": {} for col_a, col_b in edges}
    rng = np.random.default_rng(bootstrap_random_state) if bootstrap_random_state is not None else None

    for _ in range(n_iterations):
        if rng is None:
            sample_df = real_sample.sample(n=len(real_sample), replace=True)
        else:
            seed = int(rng.integers(0, np.iinfo(np.int32).max))
            sample_df = real_sample.sample(n=len(real_sample), replace=True, random_state=seed)

        for col in h_cols:
            counts = sample_df[col].value_counts(normalize=True).to_dict()
            for category, freq in counts.items():
                key = str(category)
                marginal_acc[col][key] = marginal_acc[col].get(key, 0.0) + float(freq)

        for col_a, col_b in edges:
            pair_key = f"{col_a}|{col_b}"
            joint_counts = sample_df.groupby([col_a, col_b], observed=False).size() / len(sample_df)
            for (value_a, value_b), freq in joint_counts.items():
                combo_key = f"{value_a}|{value_b}"
                joint_acc[pair_key][combo_key] = joint_acc[pair_key].get(combo_key, 0.0) + float(freq)

    final_marginals = {
        col: {key: value / n_iterations for key, value in counts.items()}
        for col, counts in marginal_acc.items()
    }
    final_joints = {
        pair_key: {key: value / n_iterations for key, value in counts.items()}
        for pair_key, counts in joint_acc.items()
    }
    return {"marginals": final_marginals, "joints": final_joints}


def build_rules_from_bootstrap_joints(joints: dict, threshold: float) -> pd.DataFrame:
    rule_rows = []
    for dependency, dist in joints.items():
        for combo, prob in dist.items():
            prob = float(prob)
            if prob <= threshold:
                rule_rows.append(
                    {
                        "Dependency": dependency,
                        "Constraint Combo": combo,
                        "Reference Probability": prob,
                        "Constraint Type": "zero" if prob == 0 else "near-zero",
                    }
                )

    if not rule_rows:
        return pd.DataFrame(columns=["Dependency", "Constraint Combo", "Reference Probability", "Constraint Type"])

    return (
        pd.DataFrame(rule_rows)
        .sort_values(["Reference Probability", "Dependency", "Constraint Combo"])
        .reset_index(drop=True)
    )


def build_near_zero_rules(
    reference_df: pd.DataFrame,
    edges: list[tuple[str, str]],
    threshold: float,
) -> pd.DataFrame:
    rule_rows = []

    for col_a, col_b in edges:
        values_a = bootstrap_categories(reference_df[col_a])
        values_b = bootstrap_categories(reference_df[col_b])
        all_combos = set(product(values_a, values_b))

        observed_probs = (
            pd.Series(list(zip(label_series(reference_df[col_a]), label_series(reference_df[col_b]))))
            .value_counts(normalize=True)
            .to_dict()
        )
        constrained_combos = []
        for value_a, value_b in all_combos:
            prob = float(observed_probs.get((value_a, value_b), 0.0))
            if prob <= threshold:
                constrained_combos.append((value_a, value_b, prob))

        constrained_combos = sorted(
            constrained_combos,
            key=lambda item: (item[2], value_sort_key(item[0]), value_sort_key(item[1])),
        )
        for value_a, value_b, prob in constrained_combos:
            rule_rows.append(
                {
                    "Dependency": f"{col_a}|{col_b}",
                    "Constraint Combo": f"{value_a}|{value_b}",
                    "Reference Probability": prob,
                    "Constraint Type": "zero" if prob == 0 else "near-zero",
                }
            )

    return pd.DataFrame(rule_rows)


def evaluate_violations(
    synthetic_df: pd.DataFrame,
    nonzero_rules: dict[tuple[str, str], set[tuple[str, str]]],
) -> tuple[pd.Series, pd.DataFrame]:
    violation_counts = pd.Series(0, index=synthetic_df.index, dtype=int)
    detail_rows = []

    for (col_a, col_b), observed_nonzero in nonzero_rules.items():
        combos = pd.Series(
            list(zip(label_series(synthetic_df[col_a]), label_series(synthetic_df[col_b]))),
            index=synthetic_df.index,
        )
        violated = ~combos.isin(observed_nonzero)
        violation_counts.loc[violated] += 1

        if violated.any():
            combo_counts = combos.loc[violated].value_counts()
            for (value_a, value_b), count in combo_counts.items():
                detail_rows.append(
                    {
                        "Dependency": f"{col_a}|{col_b}",
                        "Violation Combo": f"{value_a}|{value_b}",
                        "Count": int(count),
                    }
                )

    detail_df = pd.DataFrame(detail_rows)
    return violation_counts, detail_df


def count_rule_occurrences(dataset_name: str, df: pd.DataFrame, rule_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, rule in rule_df.iterrows():
        col_a, col_b = str(rule["Dependency"]).split("|", maxsplit=1)
        value_a, value_b = str(rule["Constraint Combo"]).split("|", maxsplit=1)
        mask = value_mask(df[col_a], value_a) & value_mask(df[col_b], value_b)
        rows.append(
            {
                "Dataset": dataset_name,
                "Dependency": rule["Dependency"],
                "Constraint Combo": rule["Constraint Combo"],
                "Reference Probability": rule["Reference Probability"],
                "Constraint Type": rule["Constraint Type"],
                "Violation Count": int(mask.sum()),
            }
        )
    return pd.DataFrame(rows)


def build_count_summary(count_tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for dataset, table in count_tables.items():
        counts = table["Violation Count"].to_numpy(dtype=float)
        rows.append(
            {
                "Dataset": dataset,
                "Constraint Count": int(len(counts)),
                "Total Violations": int(counts.sum()) if len(counts) else 0,
                "Mean Count": float(np.mean(counts)) if len(counts) else np.nan,
                "Median Count": float(np.median(counts)) if len(counts) else np.nan,
                "95th Percentile": float(np.percentile(counts, 95)) if len(counts) else np.nan,
                "Max Count": int(np.max(counts)) if len(counts) else 0,
            }
        )
    return pd.DataFrame(rows)


def format_count_summary(summary_df: pd.DataFrame) -> pd.DataFrame:
    formatted = summary_df.copy()
    for col in ["Mean Count", "Median Count", "95th Percentile"]:
        formatted[col] = formatted[col].map(lambda value: f"{value:.2f}" if pd.notna(value) else "NA")
    return formatted


def add_fit_line(
    ax: plt.Axes,
    values: np.ndarray,
    x_grid: np.ndarray,
    color: str,
    label: str,
    global_x_max: float,
) -> None:
    if len(values) == 0:
        return
    std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    value_range = float(np.max(values) - np.min(values)) if len(values) else 0.0
    if len(values) < 2 or std <= 1e-12 or value_range <= max(1.0, global_x_max * 0.02):
        return

    try:
        kde = gaussian_kde(values)
    except np.linalg.LinAlgError:
        return
    density = kde(x_grid)
    ax.plot(x_grid, density, color=color, linewidth=1.8, label=label)


def plot_violation_histograms(
    count_tables: dict[str, pd.DataFrame],
    output_path: Path,
    bins: int,
) -> None:
    colors = {
        "Real": "#1f77b4",
        "Full Model": "#ff7f0e",
        "Ablation A": "#2ca02c",
        "Ablation B": "#d62728",
        "Ablation C": "#9467bd",
    }
    all_counts = [
        table["Violation Count"].to_numpy(dtype=float)
        for table in count_tables.values()
        if not table.empty
    ]
    max_count = max((float(np.max(values)) for values in all_counts if len(values)), default=0.0)
    x_max = max(1.0, max_count * 1.05)
    bin_edges = np.linspace(0.0, x_max, bins + 1)
    x_grid = np.linspace(0.0, x_max, 400)

    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    ax_kde = ax.twinx()
    for dataset, table in count_tables.items():
        values = table["Violation Count"].to_numpy(dtype=float)
        color = colors.get(dataset, None)
        ax.hist(
            values,
            bins=bin_edges,
            density=True,
            alpha=0.22,
            color=color,
            edgecolor="none",
            label=f"{dataset} (hist)",
        )
        add_fit_line(ax_kde, values, x_grid, color or "black", f"{dataset} (KDE)", x_max)

    ax.set_title(
        "Comparison of Near-zero / Zero Constraint Violations\nHistograms with KDE-Smoothed Distributions",
        fontsize=9,
        pad=10,
    )
    ax.set_xlabel("Violation Count", fontsize=9)
    ax.set_ylabel("Histogram Density", fontsize=9)
    ax.set_ylim(0, 0.020)
    ax.tick_params(labelsize=8)
    ax_kde.set_ylabel("KDE Density", fontsize=9)
    ax_kde.set_ylim(0, 0.010)
    ax_kde.tick_params(labelsize=8)
    ax.grid(True, alpha=0.25)
    ax_kde.grid(False)
    hist_handles, hist_labels = ax.get_legend_handles_labels()
    kde_handles, kde_labels = ax_kde.get_legend_handles_labels()
    ax.legend(hist_handles + kde_handles, hist_labels + kde_labels, loc="upper right", fontsize=7, frameon=True)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def summarize_dataset(
    dataset_name: str,
    synthetic_df: pd.DataFrame,
    violation_counts: pd.Series,
    zero_rule_count: int,
) -> dict[str, object]:
    total_rows = int(len(synthetic_df))
    violation_rows = int((violation_counts > 0).sum())
    total_violations = int(violation_counts.sum())

    return {
        "Dataset": dataset_name,
        "Rows": total_rows,
        "Zero Rules": zero_rule_count,
        "Violation Rows": violation_rows,
        "Violation Rate": violation_rows / total_rows if total_rows else np.nan,
        "Total Violations": total_violations,
        "Mean Violations / Row": total_violations / total_rows if total_rows else np.nan,
        "Max Violations / Row": int(violation_counts.max()) if total_rows else 0,
    }


def build_distribution_table(distributions: dict[str, pd.Series]) -> pd.DataFrame:
    max_violations = max(int(series.max()) for series in distributions.values())
    rows = []
    for count in range(max_violations + 1):
        row = {"Violation Count Per Row": count}
        for dataset, series in distributions.items():
            n = int((series == count).sum())
            pct = n / len(series) * 100.0 if len(series) else 0.0
            row[f"{dataset} Count"] = n
            row[f"{dataset} %"] = pct
        rows.append(row)
    return pd.DataFrame(rows)


def format_summary_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    formatted = summary_df.copy()
    formatted["Violation Rate"] = formatted["Violation Rate"].map(lambda value: f"{value * 100:.2f}%")
    formatted["Mean Violations / Row"] = formatted["Mean Violations / Row"].map(lambda value: f"{value:.4f}")
    return formatted


def format_distribution_table(distribution_df: pd.DataFrame) -> pd.DataFrame:
    formatted = distribution_df.copy()
    for col in formatted.columns:
        if col.endswith(" %"):
            formatted[col] = formatted[col].map(lambda value: f"{value:.2f}%")
    return formatted


def format_top_table(detail_tables: dict[str, pd.DataFrame], top_n: int) -> pd.DataFrame:
    rows = []
    for dataset, detail_df in detail_tables.items():
        if detail_df.empty:
            continue
        ranked = detail_df.sort_values("Count", ascending=False).head(top_n)
        for _, row in ranked.iterrows():
            rows.append(
                {
                    "Dataset": dataset,
                    "Dependency": row["Dependency"],
                    "Violation Combo": row["Violation Combo"],
                    "Count": int(row["Count"]),
                }
            )
    return pd.DataFrame(rows)


def print_table(title: str, table: pd.DataFrame) -> None:
    pd.set_option("display.max_colwidth", 100)
    pd.set_option("display.width", 220)
    print()
    print(title.center(140))
    print()
    if table.empty:
        print("No rows to display.")
    else:
        print(table.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot violation-count distributions against near-zero / zero joint constraints."
    )
    parser.add_argument("--real-csv", type=Path, default=REAL_DATA_PATH, help="Path to CHARLS_processed_2020.csv.")
    parser.add_argument("--full-csv", type=Path, default=DATASET_PATHS["Full Model"], help="Path to full-model CSV.")
    parser.add_argument("--ablation-a-csv", type=Path, default=DATASET_PATHS["Ablation A"], help="Path to ablation A CSV.")
    parser.add_argument("--ablation-b-csv", type=Path, default=DATASET_PATHS["Ablation B"], help="Path to ablation B CSV.")
    parser.add_argument("--ablation-c-csv", type=Path, default=DATASET_PATHS["Ablation C"], help="Path to ablation C CSV.")
    parser.add_argument(
        "--s-target-json",
        type=Path,
        default=None,
        help="Optional JSON file containing graph bootstrap output, either {'joints': ...} or {'s_target': {'joints': ...}}.",
    )
    parser.add_argument(
        "--near-zero-threshold",
        type=float,
        default=DEFAULT_NEAR_ZERO_THRESHOLD,
        help="Reference joint probability threshold for near-zero constraints.",
    )
    parser.add_argument(
        "--edges-json",
        type=Path,
        default=None,
        help="Optional JSON list of bootstrap edges, e.g. [[\"age_bin\", \"ins\"]] or [{\"pair\": [...]}].",
    )
    parser.add_argument("--bootstrap-sample-size", type=int, default=200, help="Real rows used by graph bootstrap.")
    parser.add_argument("--sample-random-state", type=int, default=24, help="Random seed for df_real.sample.")
    parser.add_argument("--bootstrap-iterations", type=int, default=1000, help="Bootstrap iterations matching graph_v2_charls_ipf.py.")
    parser.add_argument(
        "--bootstrap-random-state",
        type=int,
        default=24,
        help="Random seed for bootstrap resampling. Use -1 for non-deterministic pandas sampling.",
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="Directory for violation distribution outputs.")
    parser.add_argument("--bins", type=int, default=12, help="Histogram bin count.")
    parser.add_argument(
        "--include-ablation-c",
        action="store_true",
        help="Also include Ablation C in the plotted comparison.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    real_df = align_columns(load_real_data(args.real_csv), "Real")

    if args.s_target_json is not None:
        s_target = load_bootstrap_target(args.s_target_json)
        bootstrap_source = f"loaded from {args.s_target_json}"
    else:
        edges = validate_edges(load_edges(args.edges_json))
        real_sample = real_df.sample(
            n=min(args.bootstrap_sample_size, len(real_df)),
            random_state=args.sample_random_state,
        ).reset_index(drop=True)
        bootstrap_seed = None if args.bootstrap_random_state == -1 else args.bootstrap_random_state
        s_target = build_bootstrap_target(
            real_sample=real_sample,
            h_cols=H_COLS,
            edges=edges,
            n_iterations=args.bootstrap_iterations,
            bootstrap_random_state=bootstrap_seed,
        )
        bootstrap_source = (
            f"computed from df_real.sample(n={len(real_sample)}, random_state={args.sample_random_state}), "
            f"iterations={args.bootstrap_iterations}"
        )

    rule_df = build_rules_from_bootstrap_joints(s_target["joints"], args.near_zero_threshold)

    datasets = {
        "Real": real_df,
        "Full Model": align_columns(load_synthetic_data(args.full_csv), "Full Model"),
        "Ablation A": align_columns(load_synthetic_data(args.ablation_a_csv), "Ablation A"),
        "Ablation B": align_columns(load_synthetic_data(args.ablation_b_csv), "Ablation B"),
        "Ablation C": align_columns(load_synthetic_data(args.ablation_c_csv), "Ablation C")
    }

    if rule_df.empty:
        print("No near-zero / zero joint combinations were found from the reference distribution.")
        return

    count_tables = {
        name: count_rule_occurrences(name, df, rule_df)
        for name, df in datasets.items()
    }
    output_path = args.output_dir / "near_zero_zero_violation_distribution.png"
    plot_violation_histograms(count_tables, output_path, bins=max(args.bins, 2))

    print(f"Reference real data: n={len(real_df)}")
    print(f"Bootstrap target source: {bootstrap_source}")
    print(f"Bootstrap joint tables: {len(s_target['joints'])}")
    print(f"Near-zero threshold: <= {args.near_zero_threshold}")
    print(f"Near-zero / zero joint combinations: {len(rule_df)}")
    print_table(
        "VIOLATION COUNT DISTRIBUTION SUMMARY",
        format_count_summary(build_count_summary(count_tables)),
    )
    print(f"\nSaved figure to: {output_path}")


if __name__ == "__main__":
    main()
