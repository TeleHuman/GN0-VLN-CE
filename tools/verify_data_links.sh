#!/bin/bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MANIFEST="${MANIFEST:-${ROOT_DIR}/data_link_manifest.txt}"

DATASET_ROOT="${DATASET_ROOT:-${ROOT_DIR}/data/datasets/R2R_VLNCE_v1-3_preprocessed}"
SCENE_DATASETS_ROOT="${SCENE_DATASETS_ROOT:-${ROOT_DIR}/data/scene_datasets}"
MP3D_ROOT="${MP3D_ROOT:-${SCENE_DATASETS_ROOT}/mp3d}"
OCCUPANCY_ROOT="${OCCUPANCY_ROOT:-${SCENE_DATASETS_ROOT}/mp3d_ce_occ}"
VERIFY_SPLITS="${VERIFY_SPLITS:-val_unseen}"

failures=0
warnings=0

mkdir -p "$(dirname "${MANIFEST}")"
exec > >(tee "${MANIFEST}") 2>&1

mark_ok() {
  echo "[OK]   $1"
}

mark_warn() {
  echo "[WARN] $1"
  warnings=$((warnings + 1))
}

mark_fail() {
  echo "[FAIL] $1"
  failures=$((failures + 1))
}

resolve_path() {
  local path="$1"
  if [[ -L "${path}" || -e "${path}" ]]; then
    readlink -f "${path}"
  else
    echo "MISSING"
  fi
}

check_dir() {
  local label="$1"
  local path="$2"
  if [[ -d "${path}" ]]; then
    mark_ok "${label}: ${path}"
  else
    mark_fail "${label} missing: ${path}"
  fi
}

check_file() {
  local label="$1"
  local path="$2"
  if [[ -f "${path}" ]]; then
    mark_ok "${label}: ${path}"
  else
    mark_fail "${label} missing: ${path}"
  fi
}

count_matches() {
  local path="$1"
  shift
  find -L "${path}" "$@" 2>/dev/null | wc -l | tr -d '[:space:]'
}

echo "Generated: $(date '+%F %T')"
echo "Root: ${ROOT_DIR}"
echo "Verify splits: ${VERIFY_SPLITS}"
echo
echo "[Resolved Paths]"
for p in \
  "${DATASET_ROOT}" \
  "${SCENE_DATASETS_ROOT}" \
  "${MP3D_ROOT}" \
  "${OCCUPANCY_ROOT}"
do
  echo "${p} -> $(resolve_path "${p}")"
done

echo
echo "[Required Directories]"
check_dir "CE dataset root" "${DATASET_ROOT}"
check_dir "Scene datasets root" "${SCENE_DATASETS_ROOT}"
check_dir "MP3D scene root" "${MP3D_ROOT}"
check_dir "CE occupancy root" "${OCCUPANCY_ROOT}"

echo
echo "[CE Episode Files]"
for split in train val_seen val_unseen; do
  check_file "${split} episode file" "${DATASET_ROOT}/${split}/${split}.json.gz"
done
if [[ -f "${DATASET_ROOT}/embeddings.json.gz" ]]; then
  mark_ok "optional embeddings file: ${DATASET_ROOT}/embeddings.json.gz"
else
  mark_warn "optional embeddings file missing: ${DATASET_ROOT}/embeddings.json.gz"
fi

echo
echo "[MP3D Scene Folders]"
if [[ -d "${MP3D_ROOT}" ]]; then
  scene_count="$(count_matches "${MP3D_ROOT}" -mindepth 1 -maxdepth 1 -type d)"
  if (( scene_count > 0 )); then
    mark_ok "found ${scene_count} MP3D scene folders"
  else
    mark_fail "no MP3D scene folders found under ${MP3D_ROOT}"
  fi

  bad_scene_count=0
  while IFS= read -r scene_dir; do
    scene_id="$(basename "${scene_dir}")"
    glb_count="$(count_matches "${scene_dir}" -maxdepth 1 -type f -name '*.glb')"
    navmesh_count="$(count_matches "${scene_dir}" -maxdepth 1 -type f -name '*.navmesh')"
    missing=()
    if (( glb_count < 1 )); then
      missing+=("*.glb")
    fi
    if (( navmesh_count < 1 )); then
      missing+=("*.navmesh")
    fi
    if (( ${#missing[@]} > 0 )); then
      mark_fail "MP3D scene ${scene_id} missing ${missing[*]}"
      bad_scene_count=$((bad_scene_count + 1))
      if (( bad_scene_count >= 20 )); then
        mark_warn "stopped listing MP3D scene file errors after 20 entries"
        break
      fi
    fi
  done < <(find -L "${MP3D_ROOT}" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort)

  if (( scene_count > 0 && bad_scene_count == 0 )); then
    mark_ok "all listed MP3D scene folders contain .glb and .navmesh files"
  fi
fi

echo
echo "[CE Occupancy Maps]"
if [[ -d "${OCCUPANCY_ROOT}" ]]; then
  occ_json_count="$(count_matches "${OCCUPANCY_ROOT}" -mindepth 2 -maxdepth 2 -type f -name 'occupancy.json')"
  occ_png_count="$(count_matches "${OCCUPANCY_ROOT}" -mindepth 2 -maxdepth 2 -type f -name 'occupancy.png')"
  if (( occ_json_count > 0 )); then
    mark_ok "found ${occ_json_count} occupancy.json files"
  else
    mark_fail "no occupancy.json files found under ${OCCUPANCY_ROOT}"
  fi
  if (( occ_png_count > 0 )); then
    mark_ok "found ${occ_png_count} occupancy.png files"
  else
    mark_fail "no occupancy.png files found under ${OCCUPANCY_ROOT}"
  fi
fi

echo
echo "[Episode Scene Cross-check]"
if command -v python >/dev/null 2>&1; then
  export DATASET_ROOT MP3D_ROOT OCCUPANCY_ROOT VERIFY_SPLITS
  set +e
  python - <<'PY'
import gzip
import json
import os
import sys
from pathlib import Path

dataset_root = Path(os.environ["DATASET_ROOT"])
mp3d_root = Path(os.environ["MP3D_ROOT"])
occupancy_root = Path(os.environ["OCCUPANCY_ROOT"])
splits = [
    split.strip()
    for split in os.environ.get("VERIFY_SPLITS", "val_unseen").split(",")
    if split.strip()
]
if not splits:
    splits = ["val_unseen"]


def scene_key(scene_id):
    if not scene_id:
        return None
    parts = str(scene_id).replace("\\", "/").split("/")
    for part in reversed(parts):
        if part.endswith((".glb", ".navmesh")):
            return Path(part).stem
    for part in reversed(parts):
        if part and part not in {"mp3d", "scene_datasets"}:
            return part
    return None


all_scene_ids = set()
parse_errors = []
for split in splits:
    path = dataset_root / split / f"{split}.json.gz"
    if not path.is_file():
        continue
    try:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception as exc:
        parse_errors.append(f"{path}: {exc}")
        continue

    episodes = payload.get("episodes", payload if isinstance(payload, list) else [])
    scene_ids = set()
    for episode in episodes:
        if not isinstance(episode, dict):
            continue
        raw_scene_id = (
            episode.get("scene_id")
            or episode.get("scene")
            or episode.get("scene_path")
        )
        key = scene_key(raw_scene_id)
        if key:
            scene_ids.add(key)
    all_scene_ids.update(scene_ids)
    print(f"[INFO] {split}: {len(episodes)} episodes, {len(scene_ids)} referenced scenes")

if parse_errors:
    for error in parse_errors[:10]:
        print(f"[FAIL] failed to parse CE episode file: {error}")
    if len(parse_errors) > 10:
        print(f"[WARN] stopped listing parse errors after 10 entries")

if not all_scene_ids:
    print("[WARN] no scene ids could be read from existing CE episode files")
    sys.exit(2 if parse_errors else 0)

mp3d_ids = {p.name for p in mp3d_root.iterdir() if p.is_dir()} if mp3d_root.is_dir() else set()
occupancy_ids = {
    p.name for p in occupancy_root.iterdir()
    if p.is_dir() and (p / "occupancy.json").is_file()
} if occupancy_root.is_dir() else set()

missing_mp3d = sorted(all_scene_ids - mp3d_ids)
missing_occupancy = sorted(all_scene_ids - occupancy_ids)

if missing_mp3d:
    print(f"[FAIL] {len(missing_mp3d)} referenced scenes are missing from MP3D root")
    for scene_id in missing_mp3d[:20]:
        print(f"       missing MP3D scene: {scene_id}")
    if len(missing_mp3d) > 20:
        print("       ...")
else:
    print(f"[OK]   all {len(all_scene_ids)} referenced scenes exist under MP3D root")

if missing_occupancy:
    print(f"[FAIL] {len(missing_occupancy)} referenced scenes are missing occupancy.json")
    for scene_id in missing_occupancy[:20]:
        print(f"       missing occupancy: {scene_id}")
    if len(missing_occupancy) > 20:
        print("       ...")
else:
    print(f"[OK]   all {len(all_scene_ids)} referenced scenes have occupancy.json")

if parse_errors or missing_mp3d or missing_occupancy:
    sys.exit(2)
PY
  py_status=$?
  set -e
  if (( py_status != 0 )); then
    failures=$((failures + 1))
  fi
else
  mark_warn "python is not available; skipped episode scene cross-check"
fi

echo
if (( failures == 0 )); then
  echo "[PASS] CE data layout looks valid. Warnings: ${warnings}"
else
  echo "[ERROR] CE data layout has ${failures} failure(s) and ${warnings} warning(s)."
fi
echo "Saved manifest: ${MANIFEST}"

if (( failures > 0 )); then
  exit 1
fi
