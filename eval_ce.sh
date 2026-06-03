#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAUNCHER="${ROOT_DIR}/habitat-tools/scripts/eval_habitat_bae_vlnce_aligned.sh"

export DISABLE_VERSION_CHECK="${DISABLE_VERSION_CHECK:-1}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/outputs/eval_ce_$(date +%Y%m%d_%H%M%S)}"
export DATASET_DATA_PATH="${DATASET_DATA_PATH:-${ROOT_DIR}/data/datasets/R2R_VLNCE_v1-3_preprocessed/val_unseen/val_unseen.json.gz}"
export SCENES_DIR="${SCENES_DIR:-${ROOT_DIR}/data/scene_datasets}"
export OCCUPANCY_ROOT="${OCCUPANCY_ROOT:-${ROOT_DIR}/data/scene_datasets/mp3d_ce_occ}"

exec bash "${LAUNCHER}" \
  --dataset-data-path "${DATASET_DATA_PATH}" \
  --scenes-dir "${SCENES_DIR}" \
  --occupancy-root "${OCCUPANCY_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  "$@"
