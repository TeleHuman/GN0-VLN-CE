#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAUNCHER="${ROOT_DIR}/habitat-tools/scripts/run_habitat_aligned_dagger_data.sh"

export DISABLE_VERSION_CHECK="${DISABLE_VERSION_CHECK:-1}"
export CE_DATA_PATH="${CE_DATA_PATH:-${ROOT_DIR}/data/datasets/R2R_VLNCE_v1-3_preprocessed/train/train.json.gz}"
export CE_OCC_ROOT="${CE_OCC_ROOT:-${ROOT_DIR}/data/scene_datasets/mp3d_ce_occ}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/outputs/dagger_ce_$(date +%Y%m%d_%H%M%S)}"

exec bash "${LAUNCHER}" \
  --ce-data-path "${CE_DATA_PATH}" \
  --ce-occ-root "${CE_OCC_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  "$@"
