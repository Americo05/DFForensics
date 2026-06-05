#!/usr/bin/env python3
"""
batch_analyze.py — Deepfake Dashboard: Batch Dataset Analyser

Walks a directory of images (flat or nested), sends each image to the
Deepfake Engine API, and saves all results to a timestamped CSV file.

Designed to be reusable across multiple datasets.

Usage:
    python batch_analyze.py <dataset_path> [OPTIONS]

Options:
    --api       API base URL            (default: http://localhost:8000)
    --out       Output CSV path         (default: results_<timestamp>.csv)
    --ext       Image extensions        (default: png,jpg,jpeg,bmp,webp)
    --limit     Max images to process   (default: all)
    --label     Ground-truth label column value, e.g. FAKE or REAL
                If omitted, label column will be empty (fill manually)
    --delay     Seconds between requests (default: 0.1, be nice to your CPU)

Examples:
    # Analyze the FaceForensics cropped_images dataset (deepfakes)
    python batch_analyze.py "/path/to/cropped_images" --label FAKE

    # Analyze a folder of real faces
    python batch_analyze.py "C:/datasets/real_faces" --label REAL --out real_results.csv

    # Quick test on first 50 images only
    python batch_analyze.py "C:/datasets/test" --limit 50
"""

import os
import sys
import csv
import time
import argparse
import requests
import threading
import concurrent.futures
from datetime import datetime
from pathlib import Path

global_paused = False

if sys.platform == "win32":
    import msvcrt
    def pause_listener():
        global global_paused
        while True:
            if msvcrt.kbhit():
                char = msvcrt.getch()
                # Use the 'p' or 'P' key to toggle pause (avoids Ctrl+C/V terminal conflicts)
                if char.lower() == b'p':
                    global_paused = not global_paused
                    if global_paused:
                        print("\n\n⏸️  Lote EM PAUSA. As conexões em curso terminam e depois para. (Pressiona a tecla P para Retomar)\n")
                    else:
                        print("\n\n▶️  Lote RETOMADO! A retomar fila...\n")
            time.sleep(0.1)
    
    threading.Thread(target=pause_listener, daemon=True).start()

# ── CLI Arguments ─────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch analyze a folder of images with the Deepfake Engine API"
    )
    parser.add_argument("dataset_path", help="Root folder containing images")
    parser.add_argument("--api",   default="http://localhost:8000", help="API base URLs (comma-separated)")
    parser.add_argument("--out",   default=None, help="Output CSV file path")
    parser.add_argument("--ext",   default="png,jpg,jpeg,bmp,webp", help="Comma-separated image extensions")
    parser.add_argument("--limit", type=int, default=None, help="Max images to process")
    parser.add_argument("--label", default="", help="Ground-truth label: FAKE or REAL (optional)")
    parser.add_argument("--delay", type=float, default=0.1, help="Delay in seconds between API calls")
    return parser.parse_args()


# ── Image Discovery ───────────────────────────────────────────────────────────

def discover_images(root: Path, extensions: list[str]) -> list[Path]:
    """Recursively find all images with the given extensions under root."""
    images = []
    for ext in extensions:
        images.extend(root.rglob(f"*.{ext}"))
        images.extend(root.rglob(f"*.{ext.upper()}"))
    images = sorted(set(images))
    return images


# ── API Call ─────────────────────────────────────────────────────────────────

# Para evitar que o Windows esgote os portos TCP virtuais ao fim de 9100 fotos (erro TIME_WAIT)
# Utiliza-se a mesma Sessão para reciclar 4 portas abertas e enviá-las continuamente
http_session = requests.Session()

def analyze_image(image_path: Path, api_url: str) -> dict | None:
    """Send a single image to /api/analyze and return the parsed JSON response."""
    endpoint = f"{api_url}/api/analyze"
    try:
        with open(image_path, "rb") as f:
            response = http_session.post(
                endpoint,
                files={"file": (image_path.name, f, "image/png")},
                timeout=60,
            )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        print(f"\n  ❌ Connection dropped on {image_path.name}. Engine might be overloaded/restarting.")
        return None
    except Exception as e:
        print(f"\n  ⚠️  Failed on {image_path.name}: {e}")
        return None


# ── Result Flattening ─────────────────────────────────────────────────────────

def flatten_result(image_path: Path, label: str, data: dict) -> dict:
    """
    Converts the API response to a flat dict suitable for a CSV row.
    Handles any number of plugins dynamically.

    New API schema (post multi-face refactor):
      data.results.overall_score         — max face score across all frames
      data.results.plugins[].name        — plugin name
      data.results.plugins[].average_score
      data.results.frame_details[].faces[].overall_score
    """
    results = data.get("results", {})
    overall_score = results.get("overall_score", None)

    # Derive decision from score (threshold: 0.6)
    if overall_score is not None:
        decision = "FAKE" if float(overall_score) >= 0.6 else "REAL"
    else:
        decision = ""

    row = {
        "image_path":        str(image_path),
        "folder":            image_path.parent.name,
        "filename":          image_path.name,
        "ground_truth":      label,
        "decision":          decision,
        "overall_score":     round(float(overall_score), 4) if overall_score is not None else "",
        "correct":           "",
    }

    # Per-plugin average scores
    for plugin in results.get("plugins", []):
        col_name = plugin["name"].replace(" ", "_").lower()
        row[f"plugin_{col_name}_avg"] = plugin.get("average_score", "")

    # Max per-face score (worst-case face in any frame)
    frame_details = results.get("frame_details", [])
    if frame_details:
        face_scores = [
            face["overall_score"]
            for frame in frame_details
            for face in frame.get("faces", [])
        ]
        row["max_face_score"] = round(max(face_scores), 4) if face_scores else ""
        row["faces_detected"] = max(len(frame.get("faces", [])) for frame in frame_details)

    # Was the prediction correct?
    if label in ("FAKE", "REAL") and decision:
        row["correct"] = "YES" if decision == label else "NO"

    return row


# ── Progress Bar ──────────────────────────────────────────────────────────────

def print_progress(current: int, total: int, label: str = "", width: int = 40):
    filled = int(width * current / total)
    bar = "█" * filled + "░" * (width - filled)
    pct = current / total * 100
    print(f"\r  [{bar}] {pct:5.1f}%  {current}/{total}  {label:<30}", end="", flush=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    root = Path(args.dataset_path)

    if not root.exists():
        print(f"❌ Dataset path does not exist: {root}")
        sys.exit(1)

    extensions = [e.strip().lstrip(".") for e in args.ext.split(",")]
    images = discover_images(root, extensions)

    if not images:
        print(f"❌ No images found in {root} with extensions: {extensions}")
        sys.exit(1)

    if args.limit:
        images = images[:args.limit]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(args.out) if args.out else Path(f"results_{timestamp}.csv")

    print(f"\n🔬 Deepfake Dashboard — Batch Analyser")
    print(f"   Dataset  : {root}")
    print(f"   Images   : {len(images)}")
    print(f"   Label    : {args.label or '(none — fill manually)'}")
    print(f"   Output   : {out_path}")
    print(f"   API      : {args.api}\n")

    # Split APIs by comma
    apis = [api.strip() for api in args.api.split(",")]
    active_apis = []

    # Quick health check
    for api in apis:
        try:
            health = requests.get(f"{api}/health", timeout=5).json()
            plugins = health.get("active_plugins", [])
            print(f"   ✅ Engine OK at {api} — {len(plugins)} plugin(s) active:")
            for p in plugins:
                print(f"      • {p.get('name', p)}")
            active_apis.append(api)
        except Exception:
            print(f"   ❌ Cannot reach engine at {api}/health.")
            
    if not active_apis:
        print("\n   ❌ No engines available. Start uvicorn first.\n")
        sys.exit(1)
    print()

    rows = []
    csv_headers_written = False
    correct_count = 0
    labeled_count = 0

    # Use semicolon as delimiter — opens correctly in European Excel
    # (PT/ES/FR locales use comma as decimal separator, semicolon as list separator)
    with open(out_path, "w", newline="", encoding="utf-8-sig") as csv_file:
        writer = None
        headers = None

        def process_image(idx, img_path):
            global global_paused
            while global_paused:
                time.sleep(0.5)

            api_url = active_apis[idx % len(active_apis)]
            data = analyze_image(img_path, api_url)
            if data is None or data.get("status") != "completed":
                return None
            return flatten_result(img_path, args.label, data)

        completed = 0
        max_threads = len(active_apis) * 4

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_threads) as executor:
            futures = {
                executor.submit(process_image, i, path): path 
                for i, path in enumerate(images)
            }
            
            try:
                for future in concurrent.futures.as_completed(futures):
                    image_path = futures[future]
                    completed += 1
                    print_progress(completed, len(images), image_path.name)
                    
                    try:
                        row = future.result()
                    except Exception as e:
                        print(f"\n⚠️  {image_path.name} failed locally: {e}")
                        row = None

                    if row:
                        if not csv_headers_written:
                            headers = list(row.keys())
                            writer = csv.DictWriter(csv_file, fieldnames=headers, delimiter=";")
                            writer.writeheader()
                            csv_headers_written = True

                        writer.writerow(row)
                        csv_file.flush()
                        rows.append(row)

                        if row["correct"] in ("YES", "NO"):
                            labeled_count += 1
                            if row["correct"] == "YES":
                                correct_count += 1

                    time.sleep(args.delay)
            except KeyboardInterrupt:
                print("\n\n⚠️ Análise do lote abortada à força (Ctrl+C). A guardar até onde chegou e ignorar as restantes...")
                executor.shutdown(wait=False, cancel_futures=True)

        # ── Summary / average row at the bottom ───────────────────────────────
        if writer and rows and headers:
            numeric_cols = [h for h in headers
                            if h not in ("image_path", "folder", "filename",
                                         "ground_truth", "decision", "correct")]
            summary = {h: "" for h in headers}
            summary["image_path"] = "=== MEDIA ==="
            summary["folder"]     = ""
            summary["filename"]   = f"{len(rows)} imagens"
            summary["ground_truth"] = args.label or "?"

            if labeled_count > 0:
                acc = correct_count / labeled_count * 100
                summary["correct"] = f"{acc:.1f}%"
            summary["decision"] = f"{correct_count}/{labeled_count} corretos" if labeled_count else ""

            for col in numeric_cols:
                vals = []
                for r in rows:
                    try:
                        vals.append(float(r.get(col, "")))
                    except (ValueError, TypeError):
                        pass
                if vals:
                    # Use comma as decimal separator for European Excel
                    summary[col] = f"{sum(vals)/len(vals):.4f}".replace(".", ",")

            # Also fix all values in data rows to use comma as decimal sep
            # (already written; this only affects summary row)
            writer.writerow(summary)

    print_progress(len(images), len(images), "Done!")
    print(f"\n\n✅ Analysis complete! Results saved to: {out_path}")
    print(f"   Total processed : {len(rows)} images")

    if labeled_count > 0:
        decisions = [r["decision"] for r in rows]
        fake_count = decisions.count("FAKE")
        real_count = decisions.count("REAL")

        # ── Confusion Matrix ──────────────────────────────────────────────
        # Ground truth is args.label (FAKE or REAL)
        # Decision is what the system predicted
        gt = args.label.upper() if args.label else ""

        if gt == "FAKE":
            tp = sum(1 for r in rows if r["decision"] == "FAKE")   # Correctly detected fakes
            fn = sum(1 for r in rows if r["decision"] == "REAL")   # Missed fakes (false negative)
            fp = 0   # No real images in this batch to misclassify
            tn = 0
        elif gt == "REAL":
            tn = sum(1 for r in rows if r["decision"] == "REAL")   # Correctly identified real
            fp = sum(1 for r in rows if r["decision"] == "FAKE")   # False alarm on real (false positive)
            tp = 0
            fn = 0
        else:
            tp = fp = tn = fn = 0

        total = tp + tn + fp + fn
        accuracy = correct_count / labeled_count * 100

        # ── Precision, Recall, F1-Score ────────────────────────────────────
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score  = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        # ── Matthews Correlation Coefficient (MCC) ────────────────────────
        # MCC = (TP*TN - FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))
        mcc_num = (tp * tn) - (fp * fn)
        mcc_den = ((tp+fp) * (tp+fn) * (tn+fp) * (tn+fn)) ** 0.5
        mcc = mcc_num / mcc_den if mcc_den > 0 else 0.0

        # ── Print Results ─────────────────────────────────────────────────
        print(f"\n{'='*56}")
        print(f"  FORENSIC METRICS SUMMARY (Ground Truth: {gt})")
        print(f"{'='*56}")
        print(f"   Accuracy        : {accuracy:.1f}%  ({correct_count}/{labeled_count})")
        print(f"   Precision       : {precision:.3f}")
        print(f"   Recall (Sens.)  : {recall:.3f}")
        print(f"   F1-Score        : {f1_score:.3f}")
        print(f"   MCC             : {mcc:.3f}")
        print(f"{'─'*56}")
        print(f"   Decided FAKE    : {fake_count}")
        print(f"   Decided REAL    : {real_count}")
        print(f"{'─'*56}")
        print(f"   CONFUSION MATRIX:")
        print(f"                     Predicted FAKE  Predicted REAL")
        print(f"   Actual FAKE    :       {tp:>5}          {fn:>5}")
        print(f"   Actual REAL    :       {fp:>5}          {tn:>5}")
        print(f"{'─'*56}")
        if gt == "FAKE":
            print(f"   TP (correct FAKE): {tp}  |  FN (missed FAKE): {fn}")
        elif gt == "REAL":
            print(f"   TN (correct REAL): {tn}  |  FP (false alarm): {fp}")
        print(f"{'='*56}")
    else:
        print(f"   (No ground-truth label provided — metrics not calculated)")
        print(f"   Tip: rerun with --label FAKE or --label REAL")

    print(f"\n   Open {out_path} in Excel or Python to review results.\n")


if __name__ == "__main__":
    main()
