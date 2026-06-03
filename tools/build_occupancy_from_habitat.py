#!/usr/bin/env python3

import argparse
import gzip
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
    import habitat_sim
except ModuleNotFoundError as exc:
    raise SystemExit(
        "habitat_sim is required. Please run with an environment that has habitat_sim "
        "(e.g. /home/lenovo/miniconda3/envs/internvla/bin/python)."
    ) from exc

try:
    import imageio.v2 as imageio
except Exception as exc:
    raise SystemExit("imageio is required to write occupancy.png") from exc


DEFAULT_CE_DATASET_ROOT = (
    "/mnt/data/GN0-VLN-CE/data/datasets/R2R_VLNCE_v1-3_preprocessed"
)
DEFAULT_SCENE_ROOT = "/mnt/data/GN0-VLN-CE/data/scene_datasets/mp3d"
DEFAULT_OUT_ROOT_CE = "/mnt/data/GN0-VLN-CE/data/scene_datasets/mp3d_ce_occ"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build occupancy.json/occupancy.png from Habitat navmesh topdown view."
    )
    parser.add_argument(
        "--task",
        choices=["ce"],
        default="ce",
        help="Which task dataset to use for scene list and floor-height estimation.",
    )
    parser.add_argument(
        "--splits",
        default="val_unseen",
        help="Comma-separated splits, e.g. val_unseen or train,val_seen,val_unseen",
    )
    parser.add_argument(
        "--meters-per-pixel",
        type=float,
        default=0.05,
        help="Topdown map resolution in meters/pixel.",
    )
    parser.add_argument(
        "--scene-root",
        default=DEFAULT_SCENE_ROOT,
        help="Source scene root containing scene navmesh and assets.",
    )
    parser.add_argument(
        "--ce-dataset-root",
        default=DEFAULT_CE_DATASET_ROOT,
        help="CE dataset root or single .json.gz file.",
    )
    parser.add_argument(
        "--ce-out-root",
        default=DEFAULT_OUT_ROOT_CE,
        help="Output scene root for CE occupancy.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing occupancy.json/png if present.",
    )
    parser.add_argument(
        "--link-scene-assets",
        action="store_true",
        help="When out-root != scene-root, symlink source scene assets into output root.",
    )
    parser.add_argument(
        "--write-bev-map",
        action="store_true",
        help="Also write a simple grayscale bev_map.png.",
    )
    parser.add_argument(
        "--limit-scenes",
        type=int,
        default=-1,
        help="Only process first N scenes after sorting.",
    )
    parser.add_argument(
        "--scene-filter",
        default="",
        help="Only process scene names containing this substring.",
    )
    parser.add_argument(
        "--report-path",
        default="",
        help="Optional report JSON path. If empty, write under out-root.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print planned actions without writing files.",
    )
    return parser.parse_args()


def _split_list(s: str) -> List[str]:
    out = [x.strip() for x in s.split(",") if x.strip()]
    return out if out else ["val_unseen"]


def _iter_split_files(dataset_root: Path, splits: Iterable[str]) -> List[Path]:
    if dataset_root.is_file():
        return [dataset_root]

    files: List[Path] = []
    for split in splits:
        candidate = dataset_root / split / f"{split}.json.gz"
        if candidate.exists():
            files.append(candidate)
    return files


def _scene_name_from_scene_id(scene_id: str) -> str:
    raw = str(scene_id).strip()
    if raw.startswith("mp3d/"):
        raw = raw[len("mp3d/") :]
    return Path(raw).stem


def _collect_scene_points(
    dataset_root: Path, splits: Iterable[str]
) -> Tuple[Dict[str, List[Tuple[float, float]]], Dict[str, List[float]]]:
    scene_points: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
    scene_heights: Dict[str, List[float]] = defaultdict(list)

    split_files = _iter_split_files(dataset_root, splits)
    if not split_files:
        return scene_points, scene_heights

    for split_file in split_files:
        with gzip.open(split_file, "rt", encoding="utf-8") as f:
            payload = json.load(f)
        episodes = payload.get("episodes", [])

        for ep in episodes:
            scene_name = _scene_name_from_scene_id(ep.get("scene_id", ""))
            if not scene_name:
                continue

            start_pos = ep.get("start_position")
            if isinstance(start_pos, list) and len(start_pos) >= 3:
                x, y, z = float(start_pos[0]), float(start_pos[1]), float(start_pos[2])
                scene_points[scene_name].append((x, z))
                scene_heights[scene_name].append(y)

            for goal in ep.get("goals", []) or []:
                pos = goal.get("position")
                if isinstance(pos, list) and len(pos) >= 3:
                    x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
                    scene_points[scene_name].append((x, z))
                    scene_heights[scene_name].append(y)

            for p in ep.get("reference_path", []) or []:
                if isinstance(p, list) and len(p) >= 3:
                    x, y, z = float(p[0]), float(p[1]), float(p[2])
                    scene_points[scene_name].append((x, z))
                    scene_heights[scene_name].append(y)

    return scene_points, scene_heights


def _pick_navmesh(scene_dir: Path, scene_name: str) -> Optional[Path]:
    direct = scene_dir / f"{scene_name}.navmesh"
    if direct.exists():
        return direct

    all_navmesh = sorted(scene_dir.rglob("*.navmesh"))
    if all_navmesh:
        return all_navmesh[0]
    return None


def _prepare_scene_out_dir(
    src_scene_dir: Path,
    out_scene_dir: Path,
    link_scene_assets: bool,
    dry_run: bool,
) -> None:
    if dry_run:
        return

    out_scene_dir.mkdir(parents=True, exist_ok=True)
    if out_scene_dir.resolve() == src_scene_dir.resolve():
        return

    if not link_scene_assets:
        return

    skip_names = {"occupancy.json", "occupancy.png", "bev_map.png"}
    for child in src_scene_dir.iterdir():
        if child.name in skip_names:
            continue
        dst = out_scene_dir / child.name
        if dst.exists() or dst.is_symlink():
            continue
        os.symlink(str(child), str(dst))


def _choose_height(
    pathfinder: "habitat_sim.PathFinder", height_hints: List[float], mpp: float
) -> Tuple[float, np.ndarray, float]:
    lb, ub = pathfinder.get_bounds()
    h_low = float(lb[1]) + 0.05
    h_high = float(ub[1]) - 0.05

    candidates: List[float] = []
    if height_hints:
        candidates.append(float(np.median(np.asarray(height_hints, dtype=np.float64))))
    candidates.extend(np.linspace(h_low, h_high, num=5).tolist())

    # Keep unique while preserving order.
    dedup: List[float] = []
    seen = set()
    for h in candidates:
        key = round(h, 4)
        if key in seen:
            continue
        seen.add(key)
        dedup.append(h)

    best_map = None
    best_h = dedup[0] if dedup else float(lb[1])
    best_ratio = -1.0

    for h in dedup:
        td = pathfinder.get_topdown_view(meters_per_pixel=mpp, height=h).astype(np.uint8)
        if td.size == 0:
            continue
        ratio = float(td.mean())
        if ratio > best_ratio:
            best_ratio = ratio
            best_h = h
            best_map = td

    if best_map is None:
        best_map = pathfinder.get_topdown_view(
            meters_per_pixel=mpp, height=float(lb[1])
        ).astype(np.uint8)
        best_ratio = float(best_map.mean()) if best_map.size > 0 else 0.0
        best_h = float(lb[1])

    return best_h, best_map, best_ratio


def _build_occupancy_payload(
    lb: np.ndarray,
    ub: np.ndarray,
    width: int,
    height: int,
    mpp: float,
    selected_height: float,
) -> dict:
    # Internal convention expected by the BAE occupancy transform functions.
    left = -float(ub[0])
    top = -float(lb[2])
    right = left + float(width) * float(mpp)
    bottom = top - float(height) * float(mpp)

    lower = [right, bottom, float(lb[1])]
    upper = [left, top, float(ub[1])]

    payload = {
        "scale": float(mpp),
        "min": [left, bottom, float(lb[1])],
        "max": [right, top, float(ub[1])],
        "lower": lower,
        "upper": upper,
        "source": "habitat_pathfinder_topdown",
        "selected_height": float(selected_height),
        "version": 1,
    }
    return payload


def _world_to_pixel_for_check(
    x: np.ndarray, z: np.ndarray, width: int, height: int, lower: List[float], upper: List[float]
) -> Tuple[np.ndarray, np.ndarray]:
    scale_x = (float(upper[0]) - float(lower[0])) / max(width, 1)
    scale_y = (float(upper[1]) - float(lower[1])) / max(height, 1)

    px = ((-x - float(lower[0])) / scale_x) - 0.5
    py = ((float(upper[1]) + z) / scale_y) - 0.5
    px = np.clip(np.rint(px).astype(np.int64), 0, width - 1)
    py = np.clip(np.rint(py).astype(np.int64), 0, height - 1)
    return px, py


def _write_task(
    task_name: str,
    dataset_root: Path,
    scene_root: Path,
    out_root: Path,
    splits: List[str],
    mpp: float,
    overwrite: bool,
    link_scene_assets: bool,
    write_bev_map: bool,
    limit_scenes: int,
    scene_filter: str,
    dry_run: bool,
) -> dict:
    points_map, heights_map = _collect_scene_points(dataset_root, splits)
    scenes = sorted(points_map.keys())
    if scene_filter:
        scenes = [s for s in scenes if scene_filter in s]
    if limit_scenes >= 0:
        scenes = scenes[:limit_scenes]

    print(
        f"[{task_name}] scenes from dataset: {len(points_map)}, selected: {len(scenes)}"
    )
    out_root.mkdir(parents=True, exist_ok=True)

    report_items = []
    for idx, scene_name in enumerate(scenes, start=1):
        src_scene_dir = scene_root / scene_name
        out_scene_dir = out_root / scene_name
        navmesh_path = _pick_navmesh(src_scene_dir, scene_name)
        if navmesh_path is None:
            report_items.append(
                {
                    "scene": scene_name,
                    "status": "skip",
                    "reason": "navmesh_not_found",
                    "src_scene_dir": str(src_scene_dir),
                }
            )
            continue

        occ_json_path = out_scene_dir / "occupancy.json"
        occ_png_path = out_scene_dir / "occupancy.png"
        bev_png_path = out_scene_dir / "bev_map.png"
        if (
            not overwrite
            and occ_json_path.exists()
            and occ_png_path.exists()
        ):
            report_items.append(
                {
                    "scene": scene_name,
                    "status": "skip",
                    "reason": "already_exists",
                    "occupancy_json": str(occ_json_path),
                    "occupancy_png": str(occ_png_path),
                }
            )
            continue

        pathfinder = habitat_sim.PathFinder()
        if not pathfinder.load_nav_mesh(str(navmesh_path)):
            report_items.append(
                {
                    "scene": scene_name,
                    "status": "skip",
                    "reason": "load_navmesh_failed",
                    "navmesh": str(navmesh_path),
                }
            )
            continue

        height_hints = heights_map.get(scene_name, [])
        selected_height, topdown, nav_ratio = _choose_height(
            pathfinder, height_hints, mpp
        )
        if topdown.size == 0:
            report_items.append(
                {
                    "scene": scene_name,
                    "status": "skip",
                    "reason": "empty_topdown",
                    "navmesh": str(navmesh_path),
                }
            )
            continue

        h, w = int(topdown.shape[0]), int(topdown.shape[1])
        lb, ub = pathfinder.get_bounds()
        occ_payload = _build_occupancy_payload(
            lb=lb,
            ub=ub,
            width=w,
            height=h,
            mpp=mpp,
            selected_height=selected_height,
        )

        occ_gray = np.where(topdown > 0, 255, 0).astype(np.uint8)

        pt_pass_ratio = None
        scene_points = points_map.get(scene_name, [])
        if scene_points:
            arr = np.asarray(scene_points, dtype=np.float64)
            px, py = _world_to_pixel_for_check(
                x=arr[:, 0],
                z=arr[:, 1],
                width=w,
                height=h,
                lower=occ_payload["lower"],
                upper=occ_payload["upper"],
            )
            pt_pass_ratio = float((occ_gray[py, px] > 200).mean())

        if not dry_run:
            _prepare_scene_out_dir(
                src_scene_dir=src_scene_dir,
                out_scene_dir=out_scene_dir,
                link_scene_assets=link_scene_assets,
                dry_run=dry_run,
            )
            out_scene_dir.mkdir(parents=True, exist_ok=True)
            imageio.imwrite(str(occ_png_path), occ_gray)
            occ_json_path.write_text(
                json.dumps(occ_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            if write_bev_map:
                bev = np.stack([occ_gray, occ_gray, occ_gray], axis=-1)
                imageio.imwrite(str(bev_png_path), bev)

        report_items.append(
            {
                "scene": scene_name,
                "status": "ok",
                "scene_index": idx,
                "navmesh": str(navmesh_path),
                "src_scene_dir": str(src_scene_dir),
                "out_scene_dir": str(out_scene_dir),
                "occupancy_json": str(occ_json_path),
                "occupancy_png": str(occ_png_path),
                "map_height": h,
                "map_width": w,
                "meters_per_pixel": float(mpp),
                "selected_height": float(selected_height),
                "topdown_navigable_ratio": float(nav_ratio),
                "point_pass_ratio": pt_pass_ratio,
            }
        )

        print(
            f"[{task_name}] [{idx}/{len(scenes)}] {scene_name} "
            f"map={h}x{w} nav={nav_ratio:.3f} points={pt_pass_ratio}"
        )

    ok_count = sum(1 for x in report_items if x["status"] == "ok")
    skip_count = len(report_items) - ok_count
    return {
        "task": task_name,
        "dataset_root": str(dataset_root),
        "scene_root": str(scene_root),
        "out_root": str(out_root),
        "splits": splits,
        "meters_per_pixel": float(mpp),
        "total_selected_scenes": len(scenes),
        "ok_scenes": ok_count,
        "skipped_scenes": skip_count,
        "items": report_items,
    }


def _default_report_path(args: argparse.Namespace) -> Path:
    task_tag = args.task
    split_tag = "_".join(_split_list(args.splits))
    return Path(args.ce_out_root) / f"occupancy_build_report_{task_tag}_{split_tag}.json"


def main() -> None:
    args = parse_args()
    splits = _split_list(args.splits)

    reports = [
        _write_task(
            task_name="ce",
            dataset_root=Path(args.ce_dataset_root),
            scene_root=Path(args.scene_root),
            out_root=Path(args.ce_out_root),
            splits=splits,
            mpp=float(args.meters_per_pixel),
            overwrite=bool(args.overwrite),
            link_scene_assets=bool(args.link_scene_assets),
            write_bev_map=bool(args.write_bev_map),
            limit_scenes=int(args.limit_scenes),
            scene_filter=str(args.scene_filter),
            dry_run=bool(args.dry_run),
        )
    ]

    output = {
        "task": args.task,
        "splits": splits,
        "meters_per_pixel": float(args.meters_per_pixel),
        "dry_run": bool(args.dry_run),
        "reports": reports,
    }

    report_path = (
        Path(args.report_path).resolve()
        if args.report_path
        else _default_report_path(args).resolve()
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[DONE] report: {report_path}")


if __name__ == "__main__":
    main()
