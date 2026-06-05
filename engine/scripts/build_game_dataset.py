"""
build_game_dataset.py — populate /public/game-images/{real,fake}/ from local datasets.

Why this exists
---------------
The /game page can show two flavours of image sets:

  • A *built-in* set that ships with the public repo (Unsplash URLs for real
    faces hard-coded in src/app/game/page.tsx; five StyleGAN sample fakes in
    public/game-images/fallback-fakes/). Small, safe to publish, runs on Vercel.

  • A *local* extended set (much larger) that the user populates from their
    own copy of FFHQ / 140k / Celeb-DF / etc. Kept off the public repo so
    the project doesn't accidentally redistribute images of real people who
    never consented to being used in a deepfake-detection challenge.

This script copies a random sample of images from a real and a fake source
folder into the local set, and writes a manifest.json so the frontend can
discover them without polling the filesystem at runtime.

Usage
-----
    cd <repo-root>
    python engine/scripts/build_game_dataset.py \
        --source-real "/path/to/140k/test/real" \
        --source-fake "/path/to/140k/test/fake" \
        --count 20

After it runs, refresh the /game page. The dashboard's `fetchManifest()`
will pick up the new images automatically and prefer them over the
built-in fallback.

Idempotent: re-running wipes the previous selection so you always get a
fresh random sample (otherwise the player would memorise the gallery).
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEST_ROOT = REPO_ROOT / "public" / "game-images"


def list_images(folder: Path) -> list[Path]:
    """All image files directly under `folder` (non-recursive)."""
    if not folder.is_dir():
        raise SystemExit(f"Source folder does not exist: {folder}")
    images = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    if not images:
        raise SystemExit(f"No image files (jpg/png/webp) found in {folder}")
    return images


def reset_dest(subdir: str) -> Path:
    """Wipe and recreate the destination subfolder so reruns start clean."""
    target = DEST_ROOT / subdir
    if target.exists():
        # Keep the .gitkeep marker if present; remove every actual image.
        for p in target.iterdir():
            if p.name == ".gitkeep":
                continue
            if p.is_file():
                p.unlink()
            elif p.is_dir():
                shutil.rmtree(p)
    else:
        target.mkdir(parents=True, exist_ok=True)
    return target


def copy_sample(images: list[Path], count: int, dest: Path, prefix: str) -> list[str]:
    """Copy `count` random images into `dest`, return manifest filenames."""
    if count > len(images):
        print(f"⚠️  Requested {count} images but source only has {len(images)} — copying all.")
        count = len(images)
    selection = random.sample(images, count)
    written = []
    for i, src in enumerate(selection, start=1):
        out_name = f"{prefix}_{i:03d}{src.suffix.lower()}"
        shutil.copy2(src, dest / out_name)
        written.append(out_name)
    return written


def write_manifest(real_files: list[str], fake_files: list[str]) -> Path:
    """Write manifest.json that the frontend fetches at game start."""
    manifest = {
        "version": 1,
        "real": [f"/game-images/real/{name}" for name in real_files],
        "fake": [f"/game-images/fake/{name}" for name in fake_files],
        "total": len(real_files) + len(fake_files),
    }
    manifest_path = DEST_ROOT / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--source-real", required=True, type=Path,
                        help="Folder with real-face images (e.g. 140k/test/real)")
    parser.add_argument("--source-fake", required=True, type=Path,
                        help="Folder with fake-face images (e.g. 140k/test/fake)")
    parser.add_argument("--count", type=int, default=20,
                        help="How many of each class to copy (default: 20)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducible selection (default: random)")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    DEST_ROOT.mkdir(parents=True, exist_ok=True)

    real_src = list_images(args.source_real)
    fake_src = list_images(args.source_fake)
    print(f"Source: {len(real_src)} real, {len(fake_src)} fake images")

    real_dest = reset_dest("real")
    fake_dest = reset_dest("fake")

    real_files = copy_sample(real_src, args.count, real_dest, "real")
    fake_files = copy_sample(fake_src, args.count, fake_dest, "fake")
    print(f"Copied: {len(real_files)} real → {real_dest}")
    print(f"        {len(fake_files)} fake → {fake_dest}")

    manifest_path = write_manifest(real_files, fake_files)
    print(f"Manifest: {manifest_path}")
    print(f"\n✅ Done. Refresh /game in the browser to see the new sample.")


if __name__ == "__main__":
    main()
