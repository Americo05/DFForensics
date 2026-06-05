"""
download_mesonet_weights.py — fetch MesoNet weights and convert to PyTorch.

What this does
--------------
The official MesoNet repo (DariusAf/MesoNet on GitHub) ships pre-trained
weights as Keras `.h5` files. The plugin in this engine uses PyTorch.
This one-shot script:

  1. Downloads `Meso4_DF.h5` from the official repository (~80 KB).
  2. Loads it with h5py (no Keras import needed — saves a dependency).
  3. Maps each Keras layer's parameters into the equivalent PyTorch
     tensor shape: conv kernels need a HWIO → OIHW transpose, BN
     parameters are kept as-is, dense layers need a transpose.
  4. Writes the resulting state_dict to engine/models/mesonet_meso4_df.pt.

After this finishes the MesoNet plugin (engine/plugins/mesonet_detector.py)
finds the weights file automatically on the next server start and
`is_configured()` flips to True.

Usage
-----
    cd <repo>
    python engine/scripts/download_mesonet_weights.py

To target a different MesoNet variant (e.g. MesoInception, or the F2F-
trained Meso4), pass `--variant`:

    python engine/scripts/download_mesonet_weights.py --variant Meso4_F2F

Why a custom converter instead of just using a community PyTorch port?
----------------------------------------------------------------------
Community ports vary in BN ordering and dense-layer transpose conventions;
loading the wrong port silently produces a model that runs and outputs
plausible numbers but with random-ish predictions. Doing the conversion
ourselves from the canonical Keras weights guarantees the inputs / outputs
match the paper.
"""

from __future__ import annotations

import argparse
import logging
import sys
import urllib.request
from pathlib import Path

logger = logging.getLogger("mesonet-download")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


# Author's canonical weights, hosted on the paper repo.
WEIGHTS_URLS: dict[str, str] = {
    "Meso4_DF":         "https://github.com/DariusAf/MesoNet/raw/master/weights/Meso4_DF.h5",
    "Meso4_F2F":        "https://github.com/DariusAf/MesoNet/raw/master/weights/Meso4_F2F.h5",
    "MesoInception_DF": "https://github.com/DariusAf/MesoNet/raw/master/weights/MesoInception_DF.h5",
    "MesoInception_F2F":"https://github.com/DariusAf/MesoNet/raw/master/weights/MesoInception_F2F.h5",
}


def _download(url: str, dest: Path) -> None:
    """Download `url` to `dest`. Streams to disk, no full-buffer in RAM."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading {url}")
    try:
        with urllib.request.urlopen(url, timeout=60) as resp, dest.open("wb") as out:
            chunk = resp.read(8192)
            total = 0
            while chunk:
                out.write(chunk)
                total += len(chunk)
                chunk = resp.read(8192)
        logger.info(f"Saved {total} bytes → {dest}")
    except Exception as e:
        # Common cause: corporate proxy, GitHub rate-limit, offline.
        sys.exit(
            f"Download failed: {e}\n\n"
            f"Manual fallback:\n"
            f"  1. Open {url} in a browser\n"
            f"  2. Save to {dest}\n"
            f"  3. Re-run this script with --skip-download"
        )


def _convert_to_pytorch(h5_path: Path, pt_path: Path, variant: str) -> None:
    """Read Keras .h5 → reshape tensors → write PyTorch state_dict."""
    try:
        import h5py
        import numpy as np
        import torch
    except ImportError as e:
        sys.exit(
            f"Missing dependency: {e.name}\n"
            "Install with: pip install h5py numpy torch"
        )

    if not variant.startswith("Meso4"):
        # MesoInception has Inception blocks (extra branches) which need a
        # different state_dict layout. The plugin only implements Meso-4.
        sys.exit(
            f"Variant '{variant}' uses MesoInception layout; the plugin only "
            "supports Meso-4 right now. Use --variant Meso4_DF or Meso4_F2F, "
            "or extend mesonet_detector.py with the inception architecture."
        )

    logger.info(f"Reading Keras weights from {h5_path}")

    # Keras .h5 has two possible layouts depending on whether the file was
    # written by `model.save()` or `model.save_weights()`. We support both
    # plus the older single-nesting variant:
    #
    #   A) save() format:
    #        model_weights/<layer>/<layer>/kernel:0
    #   B) save_weights() format (what the official MesoNet repo uses):
    #        <layer>/<layer>/kernel:0
    #   C) old Keras save_weights flat format:
    #        <layer>/kernel:0
    #
    # We auto-detect by probing for the "model_weights" group and the
    # double-nesting convention.

    state_dict: dict[str, "torch.Tensor"] = {}

    def _h5_tensor(group: "h5py.Group", path: str) -> "np.ndarray":
        return np.array(group[path])

    def _find_weight_group(layer_grp: "h5py.Group", layer_name: str) -> "h5py.Group":
        """Locate the actual kernel:0/bias:0 holder within a layer group."""
        # Doubly-nested (formats A and B): <layer>/<layer>/kernel:0
        if layer_name in layer_grp and isinstance(layer_grp[layer_name], h5py.Group):
            return layer_grp[layer_name]
        # Singly-nested (format C): <layer>/kernel:0
        return layer_grp

    with h5py.File(h5_path, "r") as f:
        root = f["model_weights"] if "model_weights" in f else f

        # Walk one level deep and collect all layer-like groups, ignoring
        # any top-level metadata datasets the Keras saver may add.
        layer_names: list[str] = [
            name for name in root.keys() if isinstance(root[name], h5py.Group)
        ]

        conv_names  = [n for n in layer_names if "conv2d" in n.lower()]
        bn_names    = [n for n in layer_names if "batch_normalization" in n.lower()]
        dense_names = [n for n in layer_names if "dense" in n.lower()]

        if len(conv_names) != 4 or len(bn_names) != 4 or len(dense_names) != 2:
            # Print the actual structure so the user can see what's wrong
            # without having to re-open the file in a separate script.
            logger.error("Unexpected Keras layer layout. Full structure:")
            def _dump(name, obj):
                kind = "Group" if isinstance(obj, h5py.Group) else f"Dataset {obj.shape}"
                logger.error(f"  {name}  [{kind}]")
            f.visititems(_dump)
            sys.exit(
                f"\nFound conv={len(conv_names)}, bn={len(bn_names)}, "
                f"dense={len(dense_names)} — expected (4, 4, 2). The model "
                "was probably re-saved with non-default layer names. Adjust "
                "the layer-name patterns above to match the dump."
            )

        # ── Convs: Keras kernel HWIO → PyTorch OIHW ──────────────────
        for i, name in enumerate(conv_names, start=1):
            group  = _find_weight_group(root[name], name)
            kernel = _h5_tensor(group, "kernel:0")            # (H, W, in, out)
            bias   = _h5_tensor(group, "bias:0")              # (out,)
            kernel_t = np.transpose(kernel, (3, 2, 0, 1))     # → (out, in, H, W)
            state_dict[f"conv{i}.weight"] = torch.from_numpy(kernel_t).float()
            state_dict[f"conv{i}.bias"]   = torch.from_numpy(bias).float()

        # ── BatchNorm: same layout in Keras and PyTorch ──────────────
        for i, name in enumerate(bn_names, start=1):
            group = _find_weight_group(root[name], name)
            state_dict[f"bn{i}.weight"]        = torch.from_numpy(_h5_tensor(group, "gamma:0")).float()
            state_dict[f"bn{i}.bias"]          = torch.from_numpy(_h5_tensor(group, "beta:0")).float()
            state_dict[f"bn{i}.running_mean"]  = torch.from_numpy(_h5_tensor(group, "moving_mean:0")).float()
            state_dict[f"bn{i}.running_var"]   = torch.from_numpy(_h5_tensor(group, "moving_variance:0")).float()
            state_dict[f"bn{i}.num_batches_tracked"] = torch.tensor(0, dtype=torch.long)

        # ── Dense: Keras kernel (in, out) → PyTorch weight (out, in) ─
        for i, name in enumerate(dense_names, start=1):
            group  = _find_weight_group(root[name], name)
            kernel = _h5_tensor(group, "kernel:0")            # (in, out)
            bias   = _h5_tensor(group, "bias:0")              # (out,)
            state_dict[f"fc{i}.weight"] = torch.from_numpy(kernel.T).float()
            state_dict[f"fc{i}.bias"]   = torch.from_numpy(bias).float()

    pt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state_dict, pt_path)
    logger.info(f"✅ Wrote PyTorch state_dict → {pt_path}  ({pt_path.stat().st_size} bytes)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--variant",
        choices=sorted(WEIGHTS_URLS.keys()),
        default="Meso4_DF",
        help="Which trained MesoNet variant to download (default: Meso4_DF — trained on FF++ Deepfakes)",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip the HTTP fetch and only run the .h5 → .pt conversion "
             "(useful if you already downloaded the file manually).",
    )
    args = parser.parse_args()

    engine_root = Path(__file__).resolve().parent.parent
    models_dir = engine_root / "models"
    h5_path = models_dir / f"{args.variant}.h5"
    pt_path = models_dir / "mesonet_meso4_df.pt"  # plugin always reads this path

    if not args.skip_download:
        _download(WEIGHTS_URLS[args.variant], h5_path)
    elif not h5_path.is_file():
        sys.exit(f"--skip-download passed but {h5_path} doesn't exist.")

    _convert_to_pytorch(h5_path, pt_path, args.variant)

    print("\nNext step:")
    print("  Restart the engine. The plugin will auto-load the weights.")
    print("  Verify with:")
    print(f"    python -c \"from engine.plugins.mesonet_detector import MesoNetDetectorPlugin; print(MesoNetDetectorPlugin().is_configured())\"")


if __name__ == "__main__":
    main()
