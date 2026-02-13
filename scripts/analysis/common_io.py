from __future__ import annotations

import glob
import os
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class InputSpec:
    out_root: str
    run_id: str
    tag: str


def default_input_glob(spec: InputSpec) -> str:
    # Matches: {out_root}/{run_id}_{label}_{sev}_{tag}/shard_*/early_exit_generic_shard*.csv
    return os.path.join(
        spec.out_root,
        f"{spec.run_id}_*_{spec.tag}",
        "shard_*",
        "early_exit_generic_shard*.csv",
    )


def load_sharded_csvs(input_glob: str) -> pd.DataFrame:
    """
    Load and concatenate shard CSVs.

    Supports:
      - A single glob pattern
      - Multiple glob patterns separated by commas
        (useful when you want to merge multiple run directories into one analysis)
    """
    globs = [g.strip() for g in str(input_glob).split(",") if g.strip()]
    if not globs:
        raise FileNotFoundError("Empty input_glob")

    # Expand each glob and keep a stable, de-duplicated list of paths.
    paths_all: list[str] = []
    for g in globs:
        paths_all.extend(sorted(glob.glob(g)))

    seen = set()
    paths: list[str] = []
    for p in paths_all:
        if p in seen:
            continue
        seen.add(p)
        paths.append(p)

    if not paths:
        raise FileNotFoundError(f"No CSVs matched glob(s): {globs}")
    frames = []
    for p in paths:
        df = pd.read_csv(p)
        df["__source_csv"] = p
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def ensure_out_dir(out_root: str, run_id: str, tag: str, subdir: str) -> str:
    out_dir = os.path.join(out_root, f"{run_id}_analysis_{tag}", subdir)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


