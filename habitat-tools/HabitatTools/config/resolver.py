from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable


def _format_split_path(path_template: str, split: str) -> Path:
    return Path(str(path_template).format(split=split))


def _to_abs(repo_root: Path, p: Path) -> Path:
    return p if p.is_absolute() else (repo_root / p)


def resolve_dataset_path(
    path_template: str,
    split: str,
    repo_root: Path,
    fallback_paths: Iterable[str] | None = None,
    validator: Callable[[Path], bool] | None = None,
) -> Path:
    candidate = _to_abs(repo_root, _format_split_path(path_template, split))
    if candidate.exists() and (validator(candidate) if validator else True):
        return candidate

    for fp in fallback_paths or []:
        alt = _to_abs(repo_root, _format_split_path(fp, split))
        if alt.exists() and (validator(alt) if validator else True):
            return alt

    return candidate


def resolve_scenes_dir(
    path_value: str,
    repo_root: Path,
    fallback_dirs: Iterable[str] | None = None,
) -> Path:
    candidate = _to_abs(repo_root, Path(path_value))
    if candidate.exists():
        return candidate

    for fd in fallback_dirs or []:
        alt = _to_abs(repo_root, Path(fd))
        if alt.exists():
            return alt

    return candidate
