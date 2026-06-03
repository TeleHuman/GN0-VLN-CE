#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/lenovo/miniconda3/envs/internvla/bin/python}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[ERROR] python not found: ${PYTHON_BIN}"
  exit 2
fi

"${PYTHON_BIN}" "${ROOT_DIR}/tools/build_occupancy_from_habitat.py" \
  --task ce \
  --scene-root "${ROOT_DIR}/data/scene_datasets/mp3d" \
  --ce-dataset-root "${ROOT_DIR}/data/datasets/R2R_VLNCE_v1-3_preprocessed" \
  --ce-out-root "${ROOT_DIR}/data/scene_datasets/mp3d_ce_occ" \
  --splits "val_unseen" \
  --meters-per-pixel 0.05 \
  --link-scene-assets \
  "$@"
