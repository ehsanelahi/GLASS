
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sensitivity_utils import PARAM_GRIDS, SUBJECT_REQS, QUALITY_SCORE_RANGE

RESULTS_DIR = "sensitivity_results"
OUTPUT_DIR = "sensitivity_plots"

SUBJECTS = ["ADAS", "RR", "UAV"]

# First three slots of the validated categorical palette (blue / orange / aqua).
SUBJECT_COLORS = {
    "ADAS": "#2a78d6",
    "RR": "#eb6834",
    "UAV": "#1baf7a",
}

PARAM_LABELS = {
    "numCounterfactuals": "Number of Counterfactuals",
    "maxIterations": "Max Iterations",
    "stepFactor": "Step Factor",
    "maxPossibilities": "Max Possibilities",
    "topK": "Top K",
    "refinementThreshold": "Confidence Threshold",
}


METRICS = [
    ("satisfaction_likelihood", "Satisfaction Likelihood"),
    ("quality_score_norm", "Quality Score (normalized to [0, 1])"),
]

FONT_SIZES = {
    "base": 14,
    "title": 17,
    "suptitle": 19,
    "label": 15,
    "tick": 13,
    "legend": 13,
}

plt.rcParams.update({
    "font.size": FONT_SIZES["base"],
    "axes.titlesize": FONT_SIZES["title"],
    "axes.labelsize": FONT_SIZES["label"],
    "xtick.labelsize": FONT_SIZES["tick"],
    "ytick.labelsize": FONT_SIZES["tick"],
    "legend.fontsize": FONT_SIZES["legend"],
})


def _load_subject_param_csv(subject, parameter_name):
    path = os.path.join(RESULTS_DIR, subject, f"{parameter_name}.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    lo, hi = QUALITY_SCORE_RANGE[subject]
    df["quality_score_norm"] = (df["quality_score"] - lo) / (hi - lo)
    return df


def _format_value_label(value):
    value = float(value)
    if value.is_integer():
        return f"{int(value):,}"
    return f"{value:g}"


def plot_parameter(parameter_name, values=None, output_dir=OUTPUT_DIR):
    if values is None:
        values = PARAM_GRIDS[parameter_name]

    loaded = {s: _load_subject_param_csv(s, parameter_name) for s in SUBJECTS}
    available = {s: df for s, df in loaded.items() if df is not None}
    if not available:
        print(f"[skip] '{parameter_name}': no CSVs found for any subject")
        return None

    missing = [s for s in SUBJECTS if s not in available]
    if missing:
        print(f"[warn] '{parameter_name}': no data for {missing}; plotting available subjects only")

    n_values = len(values)
    n_subjects = len(SUBJECTS)
    group_width = 0.8
    box_width = group_width / n_subjects
    offsets = np.linspace(-(group_width - box_width) / 2, (group_width - box_width) / 2, n_subjects)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig, axes = plt.subplots(
    1,
    len(METRICS),
    figsize=(7 * len(METRICS), 5.5)
)

    # Make axes iterable when there is only one metric
    if len(METRICS) == 1:
        axes = [axes]

    for ax, (col, metric_label) in zip(axes, METRICS):
        for subj_idx, subject in enumerate(SUBJECTS):
            if subject not in available:
                continue
            df = available[subject]

            boxes, positions = [], []
            for i, value in enumerate(values):
                subset = df[np.isclose(df["parameter_value"].astype(float), float(value))]
                boxes.append(subset[col].dropna().values)
                positions.append(i + offsets[subj_idx])

            ax.boxplot(
                boxes,
                positions=positions,
                widths=box_width * 0.9,
                patch_artist=True,
                showfliers=True,
                manage_ticks=False,
                boxprops=dict(facecolor=SUBJECT_COLORS[subject], edgecolor="black", alpha=0.85),
                medianprops=dict(color="black", linewidth=1.5),
                whiskerprops=dict(color="black"),
                capprops=dict(color="black"),
                flierprops=dict(
                    marker="o", markersize=3,
                    markerfacecolor=SUBJECT_COLORS[subject],
                    markeredgecolor="black", alpha=0.5,
                ),
            )

        ax.set_xlim(-0.5, n_values - 0.5)
        ax.set_xticks(range(n_values))
        ax.set_xticklabels([_format_value_label(v) for v in values], rotation=30, ha="right")
        ax.set_xlabel(PARAM_LABELS.get(parameter_name, parameter_name))
        ax.set_ylabel(metric_label)
        ax.set_ylim(-0.03, 1.03)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.4)
        ax.set_title(metric_label)

    legend_handles = [
        plt.Line2D([0], [0], marker="s", linestyle="", markersize=14,
                    markerfacecolor=SUBJECT_COLORS[s], markeredgecolor="black", label=s)
        for s in SUBJECTS if s in available
    ]
    fig.legend(
        handles=legend_handles, loc="upper center", ncol=len(legend_handles),
        bbox_to_anchor=(0.5, 1.06), frameon=False, fontsize=FONT_SIZES["legend"],
    )

    fig.suptitle(
        f"Sensitivity of GLASS to {PARAM_LABELS.get(parameter_name, parameter_name)}",
        y=1.14, fontsize=FONT_SIZES["suptitle"],
    )
    fig.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    png_path = os.path.join(output_dir, f"{parameter_name}.png")
    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    plt.close(fig)

    print(f"[ok] '{parameter_name}': saved {png_path}")
    return png_path


def plot_all(output_dir=OUTPUT_DIR):
    for parameter_name in PARAM_GRIDS:
        plot_parameter(parameter_name, output_dir=output_dir)


if __name__ == "__main__":

    plot_all()
