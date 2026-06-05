"""
make_benchmark_figures.py — turn the per-file CSVs into report-ready figures.

What this generates
-------------------
Reads the four per_file_scores_*.csv files from docs/benchmark_outputs/
(one per benchmark run) and produces three PNG figures suitable for the
academic report:

  1. fig_auc_heatmap.png
        Plugin × Dataset → AUC heatmap. Single glance shows which plugins
        carry signal where, and which are inverted (red = AUC < 0.5).

  2. fig_auc_per_dataset.png
        Grouped bar chart: per-plugin AUC bars for each dataset, with the
        overall ensemble AUC overlaid as a separate line. Lets the reader
        see at a glance whether the ensemble beats the best individual
        plugin (the key "ensembles add value" claim).

  3. fig_roc_overall.png
        ROC curves of the overall ensemble score for each dataset, on one
        axis. AUC numbers in the legend. Standard plot for any deepfake-
        detection paper — makes the comparison rigorous.

No pandas dependency — uses csv (stdlib) + matplotlib + numpy.

Usage
-----
    python engine/scripts/make_benchmark_figures.py

Output goes to engine/tests/benchmark_results/figures/. The benchmarks
documentation (BENCHMARKS.md / report LaTeX) references these PNGs by
relative path, so don't rename them without updating the docs.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import OrderedDict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # never pop a window — we're writing files
import matplotlib.pyplot as plt
import numpy as np


# ── Configuration ───────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# Order matters: this is the canonical reading order in the figures.
# Pairs of (csv_filename, display_label).
DATASETS: list[tuple[str, str]] = [
    ("per_file_scores_140k.csv",                 "140k (StyleGAN)"),
    ("per_file_scores_ffpp_n200.csv",            "FF++ (face-swap)"),
    ("per_file_scores_celebdf.csv",              "Celeb-DF v2 (pre-MesoNet)"),
    ("per_file_scores_celebdf_with_mesonet.csv", "Celeb-DF v2 (with MesoNet)"),
]

# Short labels used in the figures so the x-axis isn't a wall of text.
PLUGIN_COLUMNS_TO_LABELS: "OrderedDict[str, str]" = OrderedDict([
    ("plugin:DCT Frequency Analyzer",                  "DCT"),
    ("plugin:Deepfake-Specific ViT Detector",          "ViT"),
    ("plugin:Edge Blending Boundary Detector",         "Edge Blending"),
    ("plugin:MesoNet (Afchar et al., WIFS 2018)",      "MesoNet"),
    ("plugin:PRNU Noise Residue Detector",             "PRNU"),
])

OUT_DIR = REPO_ROOT / "docs" / "figures"


# ── Loaders + metrics ───────────────────────────────────────────────────

def _load_rows(csv_path: Path) -> list[dict]:
    """Read CSV into list of dicts, parsing numeric columns lazily."""
    if not csv_path.is_file():
        return []
    rows: list[dict] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def _to_pairs(rows: list[dict], score_col: str) -> tuple[list[int], list[float]]:
    """Pull (labels, scores) for a column, skipping rows with missing values."""
    labels: list[int] = []
    scores: list[float] = []
    for r in rows:
        s_raw = r.get(score_col, "")
        l_raw = r.get("label", "")
        if s_raw in (None, "", "None"):
            continue
        try:
            s = float(s_raw)
            l = int(l_raw)
        except (TypeError, ValueError):
            continue
        if np.isnan(s):
            continue
        labels.append(l)
        scores.append(s)
    return labels, scores


def _auc(labels: list[int], scores: list[float]) -> float:
    """Mann–Whitney U AUC. NaN if either class is empty."""
    pos = [s for s, l in zip(scores, labels) if l == 1]
    neg = [s for s, l in zip(scores, labels) if l == 0]
    if not pos or not neg:
        return float("nan")
    wins = ties = 0
    for p in pos:
        for n in neg:
            if p > n:
                wins += 1
            elif p == n:
                ties += 1
    return (wins + 0.5 * ties) / (len(pos) * len(neg))


def _auc_bootstrap_ci(
    labels: list[int],
    scores: list[float],
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """
    Bootstrap CI for AUC.

    Resamples (labels, scores) pairs with replacement n_boot times, computes
    AUC each time, returns (point_estimate, lower, upper) at confidence
    (1 - alpha). Default α=0.05 → 95% CI.

    With N≈200 and AUC≈0.7, the resulting half-width is typically ±0.04 to
    ±0.06 — i.e., differences smaller than that between ensemble and best
    individual plugin are NOT statistically meaningful.
    """
    if not labels or not scores:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    n = len(labels)
    point = _auc(labels, scores)
    bootstrap_aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        b_labels = [labels[i] for i in idx]
        b_scores = [scores[i] for i in idx]
        # Skip resamples that ended up single-class
        if 0 < sum(b_labels) < n:
            bootstrap_aucs.append(_auc(b_labels, b_scores))
    if not bootstrap_aucs:
        return point, float("nan"), float("nan")
    lo = float(np.percentile(bootstrap_aucs, 100 * (alpha / 2)))
    hi = float(np.percentile(bootstrap_aucs, 100 * (1 - alpha / 2)))
    return point, lo, hi


def _roc_curve(labels: list[int], scores: list[float]) -> tuple[np.ndarray, np.ndarray]:
    """ROC curve (fpr, tpr) at every unique score threshold. No sklearn."""
    pairs = sorted(zip(scores, labels), key=lambda t: -t[0])
    n_pos = sum(1 for _, l in pairs if l == 1) or 1
    n_neg = sum(1 for _, l in pairs if l == 0) or 1
    fpr_pts, tpr_pts = [0.0], [0.0]
    tp = fp = 0
    for _, l in pairs:
        if l == 1:
            tp += 1
        else:
            fp += 1
        fpr_pts.append(fp / n_neg)
        tpr_pts.append(tp / n_pos)
    return np.array(fpr_pts), np.array(tpr_pts)


# ── Figure 1: Heatmap (Plugin × Dataset) ────────────────────────────────

def figure_heatmap(dataset_rows: dict[str, list[dict]]) -> None:
    """AUC heatmap: rows = plugins (+ overall), columns = datasets."""
    plugin_cols = list(PLUGIN_COLUMNS_TO_LABELS.keys())
    plugin_lbls = list(PLUGIN_COLUMNS_TO_LABELS.values())
    rows_labels = ["Overall (ensemble)"] + plugin_lbls
    col_labels = list(dataset_rows.keys())

    matrix = np.full((len(rows_labels), len(col_labels)), np.nan)
    for j, ds_label in enumerate(col_labels):
        rows = dataset_rows[ds_label]
        # Overall is always the first row
        labels, scores = _to_pairs(rows, "overall")
        matrix[0, j] = _auc(labels, scores) if labels else np.nan
        # Then each plugin
        for i, col in enumerate(plugin_cols, start=1):
            labels, scores = _to_pairs(rows, col)
            matrix[i, j] = _auc(labels, scores) if labels else np.nan

    fig, ax = plt.subplots(figsize=(8, 5))
    # Diverging colormap centered at 0.5: red = inverted, white = random, green = useful.
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0.3, vmax=0.95, aspect="auto")

    # Cell annotations
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        color="black", fontsize=10, fontweight="bold")
            else:
                ax.text(j, i, "—", ha="center", va="center", color="gray", fontsize=10)

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=20, ha="right")
    ax.set_yticks(range(len(rows_labels)))
    ax.set_yticklabels(rows_labels)
    ax.set_title("Per-plugin AUC across datasets\n(green = useful signal, red = inverted)")

    cbar = plt.colorbar(im, ax=ax, shrink=0.7)
    cbar.set_label("AUC")
    fig.tight_layout()
    out = OUT_DIR / "fig_auc_heatmap.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out.relative_to(REPO_ROOT)}")


# ── Figure 2: Grouped bars per dataset ──────────────────────────────────

def figure_grouped_bars(dataset_rows: dict[str, list[dict]]) -> None:
    """For each dataset: per-plugin bars + overall ensemble bar."""
    plugin_cols = list(PLUGIN_COLUMNS_TO_LABELS.keys())
    plugin_lbls = list(PLUGIN_COLUMNS_TO_LABELS.values())
    ds_labels = list(dataset_rows.keys())

    n_ds = len(ds_labels)
    fig, axes = plt.subplots(1, n_ds, figsize=(4 * n_ds, 4.5), sharey=True)
    if n_ds == 1:
        axes = [axes]

    for ax, ds_label in zip(axes, ds_labels):
        rows = dataset_rows[ds_label]
        plugin_aucs: list[float] = []
        plugin_errs: list[tuple[float, float]] = []  # (lower_err, upper_err) relative to point
        for col in plugin_cols:
            labels, scores = _to_pairs(rows, col)
            if labels:
                pt, lo, hi = _auc_bootstrap_ci(labels, scores)
                plugin_aucs.append(pt)
                plugin_errs.append((pt - lo, hi - pt))
            else:
                plugin_aucs.append(float("nan"))
                plugin_errs.append((0.0, 0.0))

        labels, scores = _to_pairs(rows, "overall")
        if labels:
            overall_auc, overall_lo, overall_hi = _auc_bootstrap_ci(labels, scores)
        else:
            overall_auc = overall_lo = overall_hi = float("nan")

        x = np.arange(len(plugin_lbls))
        bar_colors = ["#2ca02c" if (not np.isnan(a) and a >= 0.5) else "#d62728"
                      for a in plugin_aucs]
        # Error bars: asymmetric (lower, upper) reflecting bootstrap 95% CI.
        err_lo = [e[0] if not np.isnan(a) else 0.0 for a, e in zip(plugin_aucs, plugin_errs)]
        err_hi = [e[1] if not np.isnan(a) else 0.0 for a, e in zip(plugin_aucs, plugin_errs)]
        ax.bar(
            x,
            [0 if np.isnan(a) else a for a in plugin_aucs],
            color=bar_colors,
            edgecolor="black",
            linewidth=0.5,
            yerr=[err_lo, err_hi],
            ecolor="#444",
            capsize=4,
        )
        # AUC values on top of each bar (above the upper error tip)
        for xi, auc, e in zip(x, plugin_aucs, plugin_errs):
            if not np.isnan(auc):
                ax.text(xi, auc + e[1] + 0.02, f"{auc:.2f}", ha="center", fontsize=8)

        # Overall as a dashed horizontal line + shaded CI band
        if not np.isnan(overall_auc):
            ax.axhline(overall_auc, color="black", linestyle="--", linewidth=1.5,
                       label=f"Ensemble = {overall_auc:.2f} [{overall_lo:.2f}, {overall_hi:.2f}]")
            ax.axhspan(overall_lo, overall_hi, color="black", alpha=0.07)
        # Random baseline
        ax.axhline(0.5, color="gray", linestyle=":", linewidth=1, label="Random (0.5)")

        ax.set_xticks(x)
        ax.set_xticklabels(plugin_lbls, rotation=30, ha="right")
        ax.set_ylim(0.0, 1.05)
        ax.set_title(ds_label, fontsize=10)
        ax.legend(loc="upper right", fontsize=7)

    axes[0].set_ylabel("AUC (95% bootstrap CI)")
    fig.suptitle(
        "Per-plugin AUC vs ensemble — bars are 95% bootstrap CIs.\n"
        "Per-plugin = simple mean aggregation. Ensemble = weighted mean + face MAX.\n"
        "Differences within overlapping CIs are not statistically meaningful.",
        fontsize=10, y=1.04,
    )
    fig.tight_layout()
    out = OUT_DIR / "fig_auc_per_dataset.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out.relative_to(REPO_ROOT)}")


# ── Figure 3: ROC curves for the ensemble ───────────────────────────────

def figure_roc_overall(dataset_rows: dict[str, list[dict]]) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for (ds_label, rows), color in zip(dataset_rows.items(), colors):
        labels, scores = _to_pairs(rows, "overall")
        if not labels:
            continue
        fpr, tpr = _roc_curve(labels, scores)
        pt, lo, hi = _auc_bootstrap_ci(labels, scores)
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f"{ds_label}: AUC = {pt:.3f}  [{lo:.3f}, {hi:.3f}]  (N={len(labels)})")

    # Diagonal = random
    ax.plot([0, 1], [0, 1], color="gray", linestyle=":", lw=1, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(
        "Ensemble ROC curves across datasets\n"
        "AUC reported with 95% bootstrap confidence intervals",
        fontsize=11,
    )
    ax.legend(loc="lower right", fontsize=8)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = OUT_DIR / "fig_roc_overall.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out.relative_to(REPO_ROOT)}")


# ── Driver ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--csv-dir",
        default=str(REPO_ROOT / "docs" / "benchmark_outputs"),
        help="Where the per_file_scores_*.csv live (default: docs/benchmark_outputs/)",
    )
    args = parser.parse_args()
    csv_dir = Path(args.csv_dir)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load every dataset CSV that actually exists. Skip silently otherwise so
    # the script still produces partial figures if the user only has some runs.
    dataset_rows: dict[str, list[dict]] = OrderedDict()
    for fname, label in DATASETS:
        path = csv_dir / fname
        rows = _load_rows(path)
        if not rows:
            print(f"  skipped: {fname} not found")
            continue
        dataset_rows[label] = rows
        print(f"  loaded {len(rows):4d} rows from {fname}")

    if len(dataset_rows) < 2:
        sys.exit("Need at least 2 datasets to generate comparison figures.")

    print("\nGenerating figures:")
    figure_heatmap(dataset_rows)
    figure_grouped_bars(dataset_rows)
    figure_roc_overall(dataset_rows)

    print(f"\nAll figures written to {OUT_DIR.relative_to(REPO_ROOT)}/")


if __name__ == "__main__":
    main()
