#!/usr/bin/env python3
"""
Randomly sample pkl files from a dataset directory and split them into train/val.

Usage:
    python sample_dataset.py
    python sample_dataset.py --dataset-dir /path/to/dataset --num-files 1000 --seed 42
"""

import random
import shutil
from dataclasses import dataclass
from pathlib import Path

import tyro


@dataclass
class Args:
    dataset_dir: Path = Path("/liujinxin/dataset/G1_motion_datasets")
    """Directory to recursively search for .pkl files."""

    num_files: int = 5000
    """Total number of .pkl files to sample."""

    output_dir: Path = Path("/liujinxin/zhaowei/G1_MOTION/data")
    """Output directory."""

    prefix: str = "pkl_5000"
    """Subfolder prefix. Output folders are <prefix>_train and <prefix>_val."""

    seed: int | None = 123
    """Random seed for reproducible sampling."""



def collect_pkl_files(dataset_dir: Path) -> list[Path]:
    pkl_files: list[Path] = []

    for path in dataset_dir.rglob("*.pkl"):
        if path.is_file():
            pkl_files.append(path)

    return sorted(pkl_files)


def split_files(files: list[Path]) -> tuple[list[Path], list[Path]]:
    if len(files) <= 1:
        return files, []

    val_count = max(1, round(len(files) * 0.01))
    train_count = len(files) - val_count
    return files[:train_count], files[train_count:]


def prepare_output_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def copy_files(files: list[Path], dst_dir: Path) -> None:
    for src in files:
        shutil.copy2(src, dst_dir / src.name)


def main() -> None:
    args = tyro.cli(Args)

    if args.num_files <= 0:
        raise ValueError("num_files must be greater than 0.")

    dataset_dir = args.dataset_dir.expanduser()
    output_dir = args.output_dir.expanduser()
    train_dir = output_dir / f"{args.prefix}_train"
    val_dir = output_dir / f"{args.prefix}_val"

    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_dir}")

    pkl_files = collect_pkl_files(dataset_dir)
    if len(pkl_files) < args.num_files:
        raise ValueError(
            f"Requested {args.num_files} files, but only found {len(pkl_files)} .pkl files."
        )

    rng = random.Random(args.seed)
    sampled_files = rng.sample(pkl_files, args.num_files)
    train_files, val_files = split_files(sampled_files)

    prepare_output_dir(train_dir)
    prepare_output_dir(val_dir)
    copy_files(train_files, train_dir)
    copy_files(val_files, val_dir)

    print(f"Found .pkl files: {len(pkl_files)}")
    print(f"Sampled files: {len(sampled_files)}")
    print(f"Train files: {len(train_files)} -> {train_dir}")
    print(f"Val files: {len(val_files)} -> {val_dir}")


if __name__ == "__main__":
    main()
