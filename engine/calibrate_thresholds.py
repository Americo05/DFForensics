"""
calibrate_thresholds.py — Optuna-driven tuning of detector thresholds.

What it does
------------
Most of the numeric breakpoints in this engine (alpha deviation, residual
std, edge ratios, lip-sync correlation buckets, the global 0.6 verdict
threshold) were initially hand-picked. This script searches for better
values against a labeled dataset, using AUC and F1 as the optimization
objectives.

How it works
------------
1. Load per-file scores from `engine/tests/benchmark_results/per_file_scores.csv`
   (produced by `engine/benchmark.py` — run that first).
2. Define an objective: given a set of trial thresholds, recompute the
   ENSEMBLE verdict for every file and return the F1 score.
3. Optuna runs `--trials` Bayesian-optimization trials and reports the
   best configuration.
4. The result is written to `engine/tests/benchmark_results/calibration.json`.
   You decide whether to commit the new values back into `verdict.ts` /
   `scene_classifier.py`.

Important: this script DOES NOT auto-patch the source code. Threshold
changes have ethical implications (false negatives in deepfake detection
can mean misinformation slips through; false positives can mean wrongly
accusing real content). A human reviews the output and decides.

Usage
-----
    # 1. Generate per-file scores first
    cd engine
    python benchmark.py --dataset ../datasets

    # 2. Calibrate against those scores
    pip install optuna
    python calibrate_thresholds.py --trials 200

Output
------
    engine/tests/benchmark_results/calibration.json
      {
        "best_f1": 0.83,
        "best_auc": 0.91,
        "thresholds": {
          "fake_verdict_threshold": 0.58,
          "audio_weight": 1.0,
          "visual_weight": 0.95,
          ...
        }
      }
"""

import argparse
import csv
import json
import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger("calibrate")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def load_scores(csv_path: Path) -> list[dict]:
    if not csv_path.is_file():
        sys.exit(
            f"Per-file scores CSV not found: {csv_path}\n"
            "Run `python engine/benchmark.py --dataset <path>` first."
        )
    rows: list[dict] = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                r["label"] = int(r["label"])
            except (KeyError, ValueError):
                continue
            for k in ("overall", "visual", "audio", "lip_sync", "wavlm"):
                try:
                    r[k] = float(r[k]) if r.get(k) not in (None, "", "None") else None
                except (TypeError, ValueError):
                    r[k] = None
            rows.append(r)
    return rows


def metrics_at(rows: list[dict], threshold: float, score_field: str = "overall") -> tuple[float, float]:
    """Return (F1, AUC) at a given binary threshold for the chosen score field."""
    tp = fp = fn = tn = 0
    pos_scores: list[float] = []
    neg_scores: list[float] = []
    for r in rows:
        s = r.get(score_field)
        if s is None:
            continue
        label = r["label"]
        if label == 1:
            pos_scores.append(s)
        else:
            neg_scores.append(s)
        pred = 1 if s > threshold else 0
        if pred == 1 and label == 1: tp += 1
        elif pred == 1 and label == 0: fp += 1
        elif pred == 0 and label == 1: fn += 1
        else: tn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall    = tp / (tp + fn) if (tp + fn) else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    # Mann-Whitney U for AUC (no sklearn)
    if not pos_scores or not neg_scores:
        auc = float("nan")
    else:
        wins = ties = 0
        for p in pos_scores:
            for n in neg_scores:
                if p > n: wins += 1
                elif p == n: ties += 1
            # break out early if too slow
        auc = (wins + 0.5 * ties) / (len(pos_scores) * len(neg_scores))
    return f1, auc


def recombine_overall(row: dict, visual_w: float, audio_w: float, agg: str) -> float | None:
    """Recompute the combined score with custom modality weights."""
    v = row.get("visual")
    a = row.get("audio")
    if v is None and a is None:
        return None
    if v is None:
        return a
    if a is None:
        return v
    if agg == "max":
        return max(v * visual_w, a * audio_w)
    # weighted mean
    total = visual_w + audio_w
    if total <= 0:
        return None
    return (v * visual_w + a * audio_w) / total


def run_optuna(rows: list[dict], n_trials: int) -> dict:
    try:
        import optuna
    except ImportError:
        sys.exit("Optuna not installed. Run: pip install optuna")

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: "optuna.Trial") -> float:
        # Search space
        threshold = trial.suggest_float("fake_verdict_threshold", 0.30, 0.85)
        visual_w  = trial.suggest_float("visual_weight", 0.5, 1.5)
        audio_w   = trial.suggest_float("audio_weight",  0.5, 1.5)
        agg       = trial.suggest_categorical("audio_visual_aggregation", ["max", "weighted_mean"])

        recomputed = [
            {**r, "overall": recombine_overall(r, visual_w, audio_w, agg)} for r in rows
        ]
        recomputed = [r for r in recomputed if r["overall"] is not None]
        if len(recomputed) < 10:
            return 0.0

        f1, _ = metrics_at(recomputed, threshold, "overall")
        return f1

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_params
    # Re-run to also capture AUC at the best params
    recomputed = [
        {**r, "overall": recombine_overall(
            r, best["visual_weight"], best["audio_weight"],
            best["audio_visual_aggregation"],
        )} for r in rows
    ]
    recomputed = [r for r in recomputed if r["overall"] is not None]
    best_f1, best_auc = metrics_at(recomputed, best["fake_verdict_threshold"], "overall")

    # Compare to the current production defaults
    current_recomputed = [
        {**r, "overall": recombine_overall(r, 1.0, 1.0, "max")}
        for r in rows
    ]
    current_recomputed = [r for r in current_recomputed if r["overall"] is not None]
    current_f1, current_auc = metrics_at(current_recomputed, 0.60, "overall")

    return {
        "best_f1": round(best_f1, 4),
        "best_auc": round(best_auc, 4),
        "current_f1": round(current_f1, 4),
        "current_auc": round(current_auc, 4),
        "improvement_f1": round(best_f1 - current_f1, 4),
        "thresholds": {
            **best,
            "fake_verdict_threshold": round(best["fake_verdict_threshold"], 4),
            "visual_weight": round(best["visual_weight"], 4),
            "audio_weight": round(best["audio_weight"], 4),
        },
        "n_trials": n_trials,
        "n_samples": len(rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune detector thresholds against benchmark scores.")
    parser.add_argument(
        "--input",
        default="tests/benchmark_results/per_file_scores.csv",
        help="CSV from engine/benchmark.py",
    )
    parser.add_argument("--trials", type=int, default=200, help="Number of Optuna trials")
    parser.add_argument(
        "--output",
        default="tests/benchmark_results/calibration.json",
        help="Where to write the best-threshold report",
    )
    args = parser.parse_args()

    base = Path(os.path.dirname(__file__))
    csv_path = base / args.input
    out_path = base / args.output

    rows = load_scores(csv_path)
    if not rows:
        sys.exit("No rows loaded — is the CSV empty?")
    n_pos = sum(1 for r in rows if r["label"] == 1)
    n_neg = sum(1 for r in rows if r["label"] == 0)
    logger.info(f"Loaded {len(rows)} samples ({n_pos} fake / {n_neg} real)")
    if n_pos < 10 or n_neg < 10:
        logger.warning("Very few samples per class — calibration will be unreliable")

    logger.info(f"Running {args.trials} Optuna trials...")
    report = run_optuna(rows, args.trials)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n=== Calibration result ===")
    print(json.dumps(report, indent=2))
    print(f"\nReport written to: {out_path}")
    print(
        "\nIMPORTANT: this does NOT auto-patch the source.\n"
        "Review the thresholds and update verdict.ts / scene_classifier.py manually.\n"
        "Sanity-check on a held-out test split before deploying."
    )


if __name__ == "__main__":
    main()
