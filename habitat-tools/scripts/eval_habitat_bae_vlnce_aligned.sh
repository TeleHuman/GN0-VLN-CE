#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HABITAT_TOOLS_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
GN0_VLN_CE_ROOT="$(cd "${HABITAT_TOOLS_ROOT}/.." && pwd)"
DEFAULT_MODEL_PATH="${GN0_VLN_CE_ROOT}/models/gn-bae-vln-ce"

PYTHON_BIN="${PYTHON_BIN:-/home/lenovo/miniconda3/envs/internutopia/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="python"
fi
PY_PREFIX="$(cd "$(dirname "${PYTHON_BIN}")/.." && pwd)"
HABITAT_SIM_EXT_DIR=""
for candidate in "${PY_PREFIX}"/lib/python*/site-packages/habitat_sim/_ext; do
  if [[ -d "${candidate}" ]]; then
    HABITAT_SIM_EXT_DIR="${candidate}"
    break
  fi
done
COMPAT_LIB_DIR="${PY_PREFIX}/compat_libs"
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
if [[ -d "${HABITAT_SIM_EXT_DIR}" ]]; then
  export LD_LIBRARY_PATH="${HABITAT_SIM_EXT_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi
if [[ -d "${COMPAT_LIB_DIR}" ]]; then
  export LD_LIBRARY_PATH="${COMPAT_LIB_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

MODEL_PATH="${MODEL_PATH:-${DEFAULT_MODEL_PATH}}"
CONFIG_PATH="${HABITAT_TOOLS_ROOT}/configs/vln_r2r_legacy.yaml"
SCENES_DIR=""
DATASET_DATA_PATH=""
CHUNKS=1
START_IDX=0
END_IDX=-1
MAX_EPISODES=0
RESUME="false"
EPISODE_KEYS_JSON=""
LOAD_DTYPE="bf16"
DEVICE="cuda:0"
ACTION_NUM=1
FALLBACK_ACTION=0
PROMPT_TYPE="V3HF"
DAGGER="false"
OCCUPANCY_ROOT="/mnt/data/GN0-VLN-CE/data/scene_datasets/mp3d_ce_occ"
ALLOW_SLIDING="true"
MODE=""
GT_STRICT_COVERAGE="false"
ASTAR_MARGINS="9,8,7,6,5,4"
SKIP_MULTIFLOOR="false"
MULTIFLOOR_HEIGHT_THRESHOLD="0.75"
OUTPUT_ROOT="${GN0_VLN_CE_ROOT}/runs/ce_aligned_gn_bae_vln_ce_$(date +%Y%m%d_%H%M%S)"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-path) MODEL_PATH="$2"; shift 2 ;;
    --config-path) CONFIG_PATH="$2"; shift 2 ;;
    --scenes-dir) SCENES_DIR="$2"; shift 2 ;;
    --dataset-data-path) DATASET_DATA_PATH="$2"; shift 2 ;;
    --chunks) CHUNKS="$2"; shift 2 ;;
    --start-idx) START_IDX="$2"; shift 2 ;;
    --end-idx) END_IDX="$2"; shift 2 ;;
    --max-episodes) MAX_EPISODES="$2"; shift 2 ;;
    --episode-keys-json) EPISODE_KEYS_JSON="$2"; shift 2 ;;
    --resume) RESUME="true"; shift 1 ;;
    --load-dtype) LOAD_DTYPE="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --action-num) ACTION_NUM="$2"; shift 2 ;;
    --fallback-action) FALLBACK_ACTION="$2"; shift 2 ;;
    --prompt-type) PROMPT_TYPE="$2"; shift 2 ;;
    --dagger) DAGGER="true"; shift 1 ;;
    --occupancy-root) OCCUPANCY_ROOT="$2"; shift 2 ;;
    --allow-sliding) ALLOW_SLIDING="true"; shift 1 ;;
    --no-allow-sliding) ALLOW_SLIDING="false"; shift 1 ;;
    --mode) MODE="$2"; shift 2 ;;
    --gt-strict-coverage) GT_STRICT_COVERAGE="true"; shift 1 ;;
    --astar-margins) ASTAR_MARGINS="$2"; shift 2 ;;
    --skip-multifloor) SKIP_MULTIFLOOR="true"; shift 1 ;;
    --multifloor-height-threshold) MULTIFLOOR_HEIGHT_THRESHOLD="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    *) echo "Unknown option: $1"; exit 2 ;;
  esac
done

mkdir -p "${OUTPUT_ROOT}"

export DISABLE_VERSION_CHECK="${DISABLE_VERSION_CHECK:-1}"
export INTERNUTOPIA_ROOT="${INTERNUTOPIA_ROOT:-/mnt/data/InternUtopia}"
export PYTHONPATH="${HABITAT_TOOLS_ROOT}:${GN0_VLN_CE_ROOT}:${INTERNUTOPIA_ROOT}:${PYTHONPATH:-}"

if command -v nvidia-smi >/dev/null 2>&1; then
  GPU_COUNT=$(nvidia-smi -L | wc -l)
else
  GPU_COUNT=1
fi
if [[ "${GPU_COUNT}" -lt 1 ]]; then
  GPU_COUNT=1
fi

trap 'echo "Terminating..."; kill 0' SIGINT SIGTERM

for ((idx=0; idx<CHUNKS; idx++)); do
  gpu_id=$(( idx % GPU_COUNT ))
  chunk_dir="${OUTPUT_ROOT}/chunk_$(printf "%03d" "${idx}")"
  mkdir -p "${chunk_dir}"

  cmd=(
    "${PYTHON_BIN}" -m HabitatTools.cli.eval_ce_aligned
    --model-path "${MODEL_PATH}"
    --output-dir "${chunk_dir}"
    --habitat-config-path "${CONFIG_PATH}"
    --split-num "${CHUNKS}"
    --split-id "${idx}"
    --start-idx "${START_IDX}"
    --end-idx "${END_IDX}"
    --max-episodes "${MAX_EPISODES}"
    --load-dtype "${LOAD_DTYPE}"
    --device "${DEVICE}"
    --action-num "${ACTION_NUM}"
    --fallback-action "${FALLBACK_ACTION}"
    --prompt-type "${PROMPT_TYPE}"
    --occupancy-root "${OCCUPANCY_ROOT}"
    --astar-margins "${ASTAR_MARGINS}"
    --multifloor-height-threshold "${MULTIFLOOR_HEIGHT_THRESHOLD}"
  )
  if [[ -n "${MODE}" ]]; then
    cmd+=(--mode "${MODE}")
  fi
  if [[ "${DAGGER}" == "true" ]]; then
    cmd+=(--dagger)
  fi
  if [[ "${ALLOW_SLIDING}" == "true" ]]; then
    cmd+=(--allow-sliding)
  elif [[ "${ALLOW_SLIDING}" == "false" ]]; then
    cmd+=(--no-allow-sliding)
  fi
  if [[ "${GT_STRICT_COVERAGE}" == "true" ]]; then
    cmd+=(--gt-strict-coverage)
  fi
  if [[ "${SKIP_MULTIFLOOR}" == "true" ]]; then
    cmd+=(--skip-multifloor)
  fi
  if [[ -n "${SCENES_DIR}" ]]; then
    cmd+=(--scenes-dir "${SCENES_DIR}")
  fi
  if [[ -n "${DATASET_DATA_PATH}" ]]; then
    cmd+=(--dataset-data-path "${DATASET_DATA_PATH}")
  fi
  if [[ -n "${EPISODE_KEYS_JSON}" ]]; then
    cmd+=(--episode-keys-json "${EPISODE_KEYS_JSON}")
  fi
  if [[ "${RESUME}" == "true" ]]; then
    cmd+=(--resume)
  fi

  CUDA_VISIBLE_DEVICES="${gpu_id}" "${cmd[@]}" &
done

wait

"${PYTHON_BIN}" -m HabitatTools.cli.merge_chunks \
  --task vlnce \
  --output-root "${OUTPUT_ROOT}" \
  --chunks "${CHUNKS}" \
  --eval-split "val_unseen"

echo "Done. Merged output: ${OUTPUT_ROOT}/merged"
