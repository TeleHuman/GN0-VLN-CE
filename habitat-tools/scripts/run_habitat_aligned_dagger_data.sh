#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HABITAT_TOOLS_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
GN0_VLN_CE_ROOT="$(cd "${HABITAT_TOOLS_ROOT}/.." && pwd)"
DEFAULT_MODEL_PATH="${GN0_VLN_CE_ROOT}/models/gn-bae-vln-ce"
CONFIG_PATH="${HABITAT_TOOLS_ROOT}/configs/vln_r2r_legacy.yaml"

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
CE_DATA_PATH="${CE_DATA_PATH:-}"
CE_EPISODE_KEYS_JSON="${CE_EPISODE_KEYS_JSON:-}"
if [[ -z "${CE_DATA_PATH}" ]]; then
  CE_DATA_PATH="/mnt/data/InternRobotics/data/vln_ce/r2r/{split}/{split}.json.gz"
fi

RUN_CE="${RUN_CE:-true}"
FULL="${FULL:-false}"
RESUME="${RESUME:-false}"
RESUME_FROM_DIR="${RESUME_FROM_DIR:-}"

START_IDX="${START_IDX:-0}"
CE_COUNT="${CE_COUNT:-1}"
CHUNKS="${CHUNKS:-1}"
CHUNK_START="${CHUNK_START:-0}"
LOCAL_CHUNKS="${LOCAL_CHUNKS:-0}"
CHUNK_PARALLELISM="${CHUNK_PARALLELISM:-1}"
CHUNK_STEALING="${CHUNK_STEALING:-false}"
CHUNK_STEAL_RETRY_SEC="${CHUNK_STEAL_RETRY_SEC:-30}"
WORKER_AUTO_RESTART="${WORKER_AUTO_RESTART:-false}"
WORKER_RESTART_DELAY_SEC="${WORKER_RESTART_DELAY_SEC:-10}"
WORKER_MAX_RESTARTS="${WORKER_MAX_RESTARTS:-0}"
LOCAL_RUN_TAG="${LOCAL_RUN_TAG:-}"
GPU_LIST="${GPU_LIST:-}"
SKIP_MERGE="${SKIP_MERGE:-false}"

PROMPT_TYPE="${PROMPT_TYPE:-V3HF}"
GT_STRICT_COVERAGE="${GT_STRICT_COVERAGE:-false}"
CE_ALLOW_SLIDING="${CE_ALLOW_SLIDING:-true}"
ASTAR_MARGINS="${ASTAR_MARGINS:-9,8,7,6,5,4}"
SKIP_MULTIFLOOR="${SKIP_MULTIFLOOR:-true}"
MULTIFLOOR_HEIGHT_THRESHOLD="${MULTIFLOOR_HEIGHT_THRESHOLD:-0.75}"

CE_OCC_ROOT="${CE_OCC_ROOT:-/mnt/data/GN0-VLN-CE/data/scene_datasets/mp3d_ce_occ}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/data/GN0-VLN-CE/dagger_data/aligned_dagger_data_${TIMESTAMP}}"
OUTPUT_ROOT_EXPLICIT="false"
REPORT_DIR=""
MANIFEST_PATH=""
PIPELINE_LOG=""
CE_OUT="${OUTPUT_ROOT}/ce_run"
CE_REPORT=""

export DISABLE_VERSION_CHECK="${DISABLE_VERSION_CHECK:-1}"
export INTERNUTOPIA_ROOT="${INTERNUTOPIA_ROOT:-/mnt/data/InternUtopia}"
export PYTHONPATH="${HABITAT_TOOLS_ROOT}:${GN0_VLN_CE_ROOT}:${INTERNUTOPIA_ROOT}:${PYTHONPATH:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-path) MODEL_PATH="$2"; shift 2 ;;
    --config-path) CONFIG_PATH="$2"; shift 2 ;;
    --ce-data-path) CE_DATA_PATH="$2"; shift 2 ;;
    --ce-episode-keys-json) CE_EPISODE_KEYS_JSON="$2"; shift 2 ;;
    --run-ce) RUN_CE="$2"; shift 2 ;;
    --full) FULL="true"; shift 1 ;;
    --resume) RESUME="true"; shift 1 ;;
    --resume-from-dir) RESUME_FROM_DIR="$2"; shift 2 ;;
    --start-idx) START_IDX="$2"; shift 2 ;;
    --ce-count) CE_COUNT="$2"; shift 2 ;;
    --chunks) CHUNKS="$2"; shift 2 ;;
    --chunk-start) CHUNK_START="$2"; shift 2 ;;
    --local-chunks) LOCAL_CHUNKS="$2"; shift 2 ;;
    --chunk-parallelism) CHUNK_PARALLELISM="$2"; shift 2 ;;
    --chunk-stealing) CHUNK_STEALING="$2"; shift 2 ;;
    --chunk-steal-retry-sec) CHUNK_STEAL_RETRY_SEC="$2"; shift 2 ;;
    --worker-auto-restart) WORKER_AUTO_RESTART="$2"; shift 2 ;;
    --worker-restart-delay-sec) WORKER_RESTART_DELAY_SEC="$2"; shift 2 ;;
    --worker-max-restarts) WORKER_MAX_RESTARTS="$2"; shift 2 ;;
    --local-run-tag) LOCAL_RUN_TAG="$2"; shift 2 ;;
    --gpu-list) GPU_LIST="$2"; shift 2 ;;
    --skip-merge) SKIP_MERGE="true"; shift 1 ;;
    --prompt-type) PROMPT_TYPE="$2"; shift 2 ;;
    --gt-strict-coverage) GT_STRICT_COVERAGE="$2"; shift 2 ;;
    --ce-allow-sliding) CE_ALLOW_SLIDING="$2"; shift 2 ;;
    --astar-margins) ASTAR_MARGINS="$2"; shift 2 ;;
    --skip-multifloor) SKIP_MULTIFLOOR="$2"; shift 2 ;;
    --multifloor-height-threshold) MULTIFLOOR_HEIGHT_THRESHOLD="$2"; shift 2 ;;
    --ce-occ-root) CE_OCC_ROOT="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; OUTPUT_ROOT_EXPLICIT="true"; shift 2 ;;
    *) echo "Unknown option: $1"; exit 2 ;;
  esac
done

if [[ "${CHUNKS}" -lt 1 ]]; then
  echo "[ERROR] --chunks must be >= 1"
  exit 2
fi
if [[ "${CHUNK_START}" -lt 0 ]]; then
  echo "[ERROR] --chunk-start must be >= 0"
  exit 2
fi
if [[ "${CHUNK_START}" -ge "${CHUNKS}" ]]; then
  echo "[ERROR] --chunk-start must be < --chunks"
  exit 2
fi

if [[ -n "${RESUME_FROM_DIR}" ]]; then
  if [[ ! -d "${RESUME_FROM_DIR}" ]]; then
    echo "[ERROR] --resume-from-dir does not exist or is not a directory: ${RESUME_FROM_DIR}"
    exit 2
  fi
  if [[ "${OUTPUT_ROOT_EXPLICIT}" == "true" && "$(readlink -f "${OUTPUT_ROOT}")" != "$(readlink -f "${RESUME_FROM_DIR}")" ]]; then
    echo "[ERROR] --resume-from-dir and --output-root point to different directories. Use only --resume-from-dir, or set --output-root to the same path."
    exit 2
  fi
  OUTPUT_ROOT="$(readlink -f "${RESUME_FROM_DIR}")"
  RESUME="true"
fi

GPU_IDS=()
if [[ -n "${GPU_LIST}" ]]; then
  IFS=',' read -r -a GPU_IDS <<< "${GPU_LIST}"
else
  if command -v nvidia-smi >/dev/null 2>&1; then
    mapfile -t GPU_IDS < <(nvidia-smi --query-gpu=index --format=csv,noheader | awk '{print $1}')
  fi
fi
if [[ "${#GPU_IDS[@]}" -eq 0 ]]; then
  GPU_IDS=("0")
fi

if [[ "${LOCAL_CHUNKS}" -le 0 ]]; then
  LOCAL_CHUNKS="${#GPU_IDS[@]}"
fi
if [[ "${LOCAL_CHUNKS}" -lt 1 ]]; then
  echo "[ERROR] --local-chunks must be >= 1"
  exit 2
fi
if [[ "${CHUNK_PARALLELISM}" -lt 1 ]]; then
  echo "[ERROR] --chunk-parallelism must be >= 1"
  exit 2
fi
if [[ "${CHUNK_STEALING}" == "true" && "${CHUNK_PARALLELISM}" -ne 1 ]]; then
  echo "[ERROR] --chunk-stealing true currently requires --chunk-parallelism 1"
  exit 2
fi
if [[ "${WORKER_MAX_RESTARTS}" -lt 0 ]]; then
  echo "[ERROR] --worker-max-restarts must be >= 0 (0 means unlimited)"
  exit 2
fi
if [[ "${LOCAL_CHUNKS}" -gt "${#GPU_IDS[@]}" ]]; then
  echo "[ERROR] local chunk count ${LOCAL_CHUNKS} exceeds available gpu slots ${#GPU_IDS[@]}"
  exit 2
fi
if [[ $((CHUNK_START + LOCAL_CHUNKS)) -gt "${CHUNKS}" ]]; then
  echo "[ERROR] local chunk window [${CHUNK_START}, $((CHUNK_START + LOCAL_CHUNKS))) exceeds global chunks ${CHUNKS}"
  exit 2
fi

LOCAL_SPLIT_IDS=()
for ((local_rank=0; local_rank<LOCAL_CHUNKS; local_rank++)); do
  LOCAL_SPLIT_IDS+=("$((CHUNK_START + local_rank))")
done
LOCAL_SPLIT_IDS_CSV="$(IFS=,; echo "${LOCAL_SPLIT_IDS[*]}")"
GPU_LIST_RESOLVED="$(IFS=,; echo "${GPU_IDS[*]}")"
if [[ -z "${LOCAL_RUN_TAG}" ]]; then
  LOCAL_RUN_TAG="$(hostname)_chunks_${CHUNK_START}_$((CHUNK_START + LOCAL_CHUNKS - 1))"
fi

REPORT_DIR="${OUTPUT_ROOT}/reports/${LOCAL_RUN_TAG}"
MANIFEST_PATH="${OUTPUT_ROOT}/run_manifest_${LOCAL_RUN_TAG}.json"
PIPELINE_LOG="${OUTPUT_ROOT}/pipeline_${LOCAL_RUN_TAG}.log"
CE_OUT="${OUTPUT_ROOT}/ce_run"
CE_REPORT="${REPORT_DIR}/dagger_ce_report.json"

mkdir -p "${OUTPUT_ROOT}" "${REPORT_DIR}"
exec > >(tee -a "${PIPELINE_LOG}") 2>&1

if [[ "${RUN_CE}" != "true" ]]; then
  echo "[ERROR] CE must be enabled in the submit branch."
  exit 2
fi

if [[ ! -e "${MODEL_PATH}" ]]; then
  echo "[ERROR] model path not found: ${MODEL_PATH}"
  exit 2
fi
if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "[ERROR] habitat config not found: ${CONFIG_PATH}"
  exit 2
fi
if [[ ! -d "${CE_OCC_ROOT}" && "${RUN_CE}" == "true" ]]; then
  echo "[ERROR] CE occupancy root not found: ${CE_OCC_ROOT}"
  exit 2
fi
if [[ -n "${CE_EPISODE_KEYS_JSON}" && ! -f "${CE_EPISODE_KEYS_JSON}" ]]; then
  echo "[ERROR] CE episode key filter not found: ${CE_EPISODE_KEYS_JSON}"
  exit 2
fi

echo "[INFO] output_root=${OUTPUT_ROOT}"
echo "[INFO] model_path=${MODEL_PATH}"
echo "[INFO] run_ce=${RUN_CE} full=${FULL}"
echo "[INFO] start_idx=${START_IDX} ce_count=${CE_COUNT}"
echo "[INFO] chunks=${CHUNKS} chunk_start=${CHUNK_START} local_chunks=${LOCAL_CHUNKS} chunk_parallelism=${CHUNK_PARALLELISM}"
echo "[INFO] chunk_stealing=${CHUNK_STEALING} chunk_steal_retry_sec=${CHUNK_STEAL_RETRY_SEC}"
echo "[INFO] worker_auto_restart=${WORKER_AUTO_RESTART} worker_restart_delay_sec=${WORKER_RESTART_DELAY_SEC} worker_max_restarts=${WORKER_MAX_RESTARTS}"
echo "[INFO] local_split_ids=${LOCAL_SPLIT_IDS_CSV}"
echo "[INFO] local_run_tag=${LOCAL_RUN_TAG}"
echo "[INFO] gpu_list=${GPU_LIST_RESOLVED}"
echo "[INFO] resume=${RESUME} resume_from_dir=${RESUME_FROM_DIR} skip_merge=${SKIP_MERGE}"
echo "[INFO] astar_margins=${ASTAR_MARGINS}"
echo "[INFO] skip_multifloor=${SKIP_MULTIFLOOR} multifloor_height_threshold=${MULTIFLOOR_HEIGHT_THRESHOLD}"
echo "[INFO] ce_data_path=${CE_DATA_PATH}"
if [[ -n "${CE_EPISODE_KEYS_JSON}" ]]; then
  echo "[INFO] ce_episode_keys_json=${CE_EPISODE_KEYS_JSON}"
fi
if [[ "${GT_STRICT_COVERAGE}" == "true" ]]; then
  echo "[INFO] dagger_action_policy=gt_override_when_available"
else
  echo "[INFO] dagger_action_policy=vlm_action_execution"
fi

"${PYTHON_BIN}" - <<PY
import json
from pathlib import Path

payload = {
  "engine": "habitat_tools_aligned_dagger_data",
  "model_path": "${MODEL_PATH}",
  "config_path": "${CONFIG_PATH}",
  "run_ce": "${RUN_CE}" == "true",
  "full": "${FULL}" == "true",
  "resume": "${RESUME}" == "true",
  "resume_from_dir": "${RESUME_FROM_DIR}",
  "start_idx": int("${START_IDX}"),
  "ce_count": int("${CE_COUNT}"),
  "chunks": int("${CHUNKS}"),
  "chunk_start": int("${CHUNK_START}"),
  "local_chunks": int("${LOCAL_CHUNKS}"),
  "chunk_parallelism": int("${CHUNK_PARALLELISM}"),
  "chunk_stealing": "${CHUNK_STEALING}" == "true",
  "chunk_steal_retry_sec": float("${CHUNK_STEAL_RETRY_SEC}"),
  "worker_auto_restart": "${WORKER_AUTO_RESTART}" == "true",
  "worker_restart_delay_sec": float("${WORKER_RESTART_DELAY_SEC}"),
  "worker_max_restarts": int("${WORKER_MAX_RESTARTS}"),
  "local_run_tag": "${LOCAL_RUN_TAG}",
  "local_split_ids": [int(x) for x in "${LOCAL_SPLIT_IDS_CSV}".split(",") if x],
  "gpu_list": [x for x in "${GPU_LIST_RESOLVED}".split(",") if x],
  "prompt_type": "${PROMPT_TYPE}",
  "gt_strict_coverage": "${GT_STRICT_COVERAGE}" == "true",
  "ce_allow_sliding": "${CE_ALLOW_SLIDING}" == "true",
  "astar_margins": "${ASTAR_MARGINS}",
  "skip_multifloor": "${SKIP_MULTIFLOOR}" == "true",
  "multifloor_height_threshold": float("${MULTIFLOOR_HEIGHT_THRESHOLD}"),
  "ce_data_path": "${CE_DATA_PATH}",
  "ce_episode_keys_json": "${CE_EPISODE_KEYS_JSON}",
  "ce_occ_root": "${CE_OCC_ROOT}",
  "output_root": "${OUTPUT_ROOT}",
  "skip_merge": "${SKIP_MERGE}" == "true",
}
Path("${MANIFEST_PATH}").write_text(
  json.dumps(payload, ensure_ascii=False, indent=2),
  encoding="utf-8",
)
print(f"[INFO] wrote manifest: ${MANIFEST_PATH}")
PY

run_check() {
  local task="$1"
  local run_path="$2"
  local out="$3"
  local expected="${4:-}"

  local cmd=(
    "${PYTHON_BIN}" "${GN0_VLN_CE_ROOT}/tools/check_dagger_samples.py"
    --run-path "${run_path}"
    --task "${task}"
    --out "${out}"
  )
  if [[ -n "${expected}" ]]; then
    cmd+=(--expect-episodes "${expected}")
  fi
  if [[ "${GT_STRICT_COVERAGE}" != "true" ]]; then
    cmd+=(--require-gt-rollout-valid-rate 0.0)
  fi
  "${cmd[@]}"
}

aggregate_reports() {
  local task="$1"
  local summary_out="$2"
  shift 2
  "${PYTHON_BIN}" - "$task" "$summary_out" "$@" <<'PY'
import json
import sys
from pathlib import Path

task = sys.argv[1]
summary_out = Path(sys.argv[2])
report_paths = [Path(p) for p in sys.argv[3:]]
items = []
passed = 0
failed = 0
for path in report_paths:
    obj = json.loads(path.read_text(encoding="utf-8"))
    status = str(obj.get("status", "unknown"))
    items.append({
        "path": str(path),
        "run_path": obj.get("run_path"),
        "status": status,
        "episode_count": obj.get("episode_count"),
    })
    if status == "pass":
        passed += 1
    else:
        failed += 1

summary = {
    "task": task,
    "local_chunk_reports": items,
    "local_chunk_count": len(items),
    "pass_count": passed,
    "fail_count": failed,
    "status": "pass" if failed == 0 else "fail",
}
summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"[INFO] wrote summary: {summary_out}")
PY
}

maybe_merge_task() {
  local task="$1"
  local output_root="$2"

  if [[ "${SKIP_MERGE}" == "true" ]]; then
    echo "[INFO] skip merge for ${task}: requested by --skip-merge"
    return 0
  fi

  local missing=0
  for ((idx=0; idx<CHUNKS; idx++)); do
    local marker="${output_root}/chunk_$(printf "%03d" "${idx}")/chunk.done"
    if [[ ! -f "${marker}" ]]; then
      missing=1
      break
    fi
  done
  if [[ "${missing}" == "1" ]]; then
    echo "[INFO] merge for ${task} deferred: waiting for all ${CHUNKS} chunk.done markers under ${output_root}"
    return 0
  fi

  echo "[INFO] merging ${task} chunks under ${output_root}"
  "${PYTHON_BIN}" -m HabitatTools.cli.merge_chunks \
    --task "${task}" \
    --output-root "${output_root}" \
    --chunks "${CHUNKS}" \
    --eval-split "val_unseen"
}

mark_chunk_done_if_complete() {
  local chunk_dir="$1"
  "${PYTHON_BIN}" - "${chunk_dir}" <<'PY'
import json
import sys
from pathlib import Path

chunk_dir = Path(sys.argv[1])
manifest = chunk_dir / ".queue" / "episodes.jsonl"
done_dir = chunk_dir / ".queue" / "done"
marker = chunk_dir / "chunk.done"

if not manifest.exists():
    raise SystemExit(1)

total = 0
missing = 0
with manifest.open("r", encoding="utf-8") as handle:
    for line in handle:
        line = line.strip()
        if not line:
            continue
        total += 1
        item = json.loads(line)
        scene_id = str(item["scene_id"]).replace("/", "__")
        episode_id = int(item["episode_id"])
        done_marker = done_dir / f"{scene_id}_{episode_id:04d}.done"
        if not done_marker.exists():
            missing += 1

if total > 0 and missing == 0:
    marker.write_text("done\n", encoding="utf-8")
    raise SystemExit(0)

raise SystemExit(1)
PY
}

chunk_has_claimable_work() {
  local chunk_dir="$1"
  "${PYTHON_BIN}" - "${chunk_dir}" <<'PY'
import json
import os
import socket
import sys
import time
from pathlib import Path

chunk_dir = Path(sys.argv[1])
if (chunk_dir / "chunk.done").exists():
    raise SystemExit(1)

manifest = chunk_dir / ".queue" / "episodes.jsonl"
if not manifest.exists():
    # No queue yet. A worker should initialize this chunk.
    raise SystemExit(0)

done_dir = chunk_dir / ".queue" / "done"
claims_dir = chunk_dir / ".queue" / "claims"
current_host = socket.gethostname()
stale_sec = float(os.environ.get("HABITAT_TOOLS_CLAIM_STALE_SEC", "86400"))

def active_claim(path: Path) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    claim_host = str(payload.get("host") or "")
    if claim_host and claim_host != current_host:
        try:
            return (time.time() - path.stat().st_mtime) < stale_sec
        except OSError:
            return False
    try:
        pid = int(payload.get("pid"))
    except Exception:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True

total = 0
all_done = True
with manifest.open("r", encoding="utf-8") as handle:
    for line in handle:
        line = line.strip()
        if not line:
            continue
        total += 1
        item = json.loads(line)
        scene_id = str(item["scene_id"]).replace("/", "__")
        episode_id = int(item["episode_id"])
        marker = f"{scene_id}_{episode_id:04d}"
        if (done_dir / f"{marker}.done").exists():
            continue
        all_done = False
        claim_path = claims_dir / f"{marker}.claim"
        if not claim_path.exists() or not active_claim(claim_path):
            raise SystemExit(0)

if total > 0 and all_done:
    (chunk_dir / "chunk.done").write_text("done\n", encoding="utf-8")

raise SystemExit(1)
PY
}

all_task_chunks_done() {
  local output_root="$1"
  local idx
  for ((idx=0; idx<CHUNKS; idx++)); do
    if [[ ! -f "${output_root}/chunk_$(printf "%03d" "${idx}")/chunk.done" ]]; then
      return 1
    fi
  done
  return 0
}

emit_steal_split_ids() {
  local idx
  for ((idx=CHUNKS-1; idx>=0; idx--)); do
    echo "${idx}"
  done
}

launch_task() {
  local task_label="$1"
  local task_name="$2"
  local cli_module="$3"
  local dataset_path="$4"
  local episode_keys_json="$5"
  local occupancy_root="$6"
  local allow_sliding="$7"
  local output_root="$8"
  local summary_report="$9"
  local expected_count="${10}"

  local end_idx="-1"
  if [[ "${FULL}" != "true" ]]; then
    end_idx="$((START_IDX + expected_count))"
  fi

  local report_subdir="${REPORT_DIR}/${task_label}"
  mkdir -p "${output_root}" "${report_subdir}"

  echo "[INFO] running ${task_label} dagger collect..."
  echo "[INFO] ${task_label} output_root=${output_root}"

  if [[ "${CHUNK_STEALING}" == "true" ]]; then
    local -a pids=()
    local -a report_paths=()

    run_steal_split() {
      local split_id="$1"
      local gpu_id="$2"
      local slot_id="$3"
      local role="$4"
      local chunk_dir="${output_root}/chunk_$(printf "%03d" "${split_id}")"
      local chunk_log="${chunk_dir}/runner.log"
      local steal_log="${chunk_dir}/runner_steal_slot_$(printf "%02d" "${slot_id}").log"
      mkdir -p "${chunk_dir}"

      if [[ -f "${chunk_dir}/chunk.done" ]]; then
        echo "[INFO][${task_label}] ${role} skip already-complete split_id=${split_id} gpu=${gpu_id}" | tee -a "${steal_log}"
        return 0
      fi

      local -a cmd=(
        "${PYTHON_BIN}" -m "${cli_module}"
        --model-path "${MODEL_PATH}"
        --output-dir "${chunk_dir}"
        --habitat-config-path "${CONFIG_PATH}"
        --dataset-data-path "${dataset_path}"
        --occupancy-root "${occupancy_root}"
        --prompt-type "${PROMPT_TYPE}"
        --astar-margins "${ASTAR_MARGINS}"
        --multifloor-height-threshold "${MULTIFLOOR_HEIGHT_THRESHOLD}"
        --split-num "${CHUNKS}"
        --split-id "${split_id}"
        --start-idx "${START_IDX}"
        --end-idx "${end_idx}"
        --mode collect
        --queue-worker-id "${slot_id}"
        --queue-worker-count "${CHUNKS}"
      )
      cmd+=(--dagger)
      if [[ -n "${episode_keys_json}" ]]; then
        cmd+=(--episode-keys-json "${episode_keys_json}")
      fi
      cmd+=(--resume)
      if [[ "${GT_STRICT_COVERAGE}" == "true" ]]; then
        cmd+=(--gt-strict-coverage)
      fi
      if [[ "${allow_sliding}" == "true" ]]; then
        cmd+=(--allow-sliding)
      elif [[ "${allow_sliding}" == "false" ]]; then
        cmd+=(--no-allow-sliding)
      fi
      if [[ "${SKIP_MULTIFLOOR}" == "true" ]]; then
        cmd+=(--skip-multifloor)
      fi

      local restart_count=0
      local attempt=1
      local worker_status=0
      while true; do
        echo "[INFO][${task_label}] ${role}_launch split_id=${split_id} slot=${slot_id} gpu=${gpu_id} attempt=${attempt} restart_count=${restart_count}" | tee -a "${chunk_log}" "${steal_log}"
        set +e
        CUDA_VISIBLE_DEVICES="${gpu_id}" "${cmd[@]}" 2>&1 | tee -a "${chunk_log}" "${steal_log}"
        worker_status="${PIPESTATUS[0]}"
        set -e
        echo "[INFO][${task_label}] ${role}_exit split_id=${split_id} slot=${slot_id} status=${worker_status} attempt=${attempt} restart_count=${restart_count}" | tee -a "${chunk_log}" "${steal_log}"
        if [[ "${worker_status}" == "0" ]]; then
          break
        fi
        if [[ "${WORKER_AUTO_RESTART}" != "true" ]]; then
          return "${worker_status}"
        fi
        restart_count=$((restart_count + 1))
        if [[ "${WORKER_MAX_RESTARTS}" -gt 0 && "${restart_count}" -gt "${WORKER_MAX_RESTARTS}" ]]; then
          echo "[ERROR][${task_label}] ${role}_restart_exhausted split_id=${split_id} slot=${slot_id} status=${worker_status} restart_count=${restart_count}" | tee -a "${chunk_log}" "${steal_log}"
          return "${worker_status}"
        fi
        echo "[WARN][${task_label}] ${role}_restart split_id=${split_id} slot=${slot_id} status=${worker_status} restart_count=${restart_count} sleep_sec=${WORKER_RESTART_DELAY_SEC}" | tee -a "${chunk_log}" "${steal_log}"
        sleep "${WORKER_RESTART_DELAY_SEC}"
        attempt=$((attempt + 1))
      done

      if mark_chunk_done_if_complete "${chunk_dir}"; then
        echo "[INFO][${task_label}] marked complete split_id=${split_id}" | tee -a "${chunk_log}" "${steal_log}"
      fi
      return 0
    }

    for ((local_rank=0; local_rank<LOCAL_CHUNKS; local_rank++)); do
      (
        set -euo pipefail
        local initial_split_id="${LOCAL_SPLIT_IDS[${local_rank}]}"
        local gpu_id="${GPU_IDS[${local_rank}]}"
        local slot_id="${local_rank}"

        run_steal_split "${initial_split_id}" "${gpu_id}" "${slot_id}" "primary"

        while true; do
          if all_task_chunks_done "${output_root}"; then
            echo "[INFO][${task_label}] steal slot=${slot_id} all chunks complete"
            break
          fi

          local attempted=0
          local split_id
          while IFS= read -r split_id; do
            local chunk_dir="${output_root}/chunk_$(printf "%03d" "${split_id}")"
            if [[ -f "${chunk_dir}/chunk.done" ]]; then
              continue
            fi
            if ! chunk_has_claimable_work "${chunk_dir}"; then
              continue
            fi
            attempted=1
            echo "[INFO][${task_label}] steal slot=${slot_id} gpu=${gpu_id} trying split_id=${split_id}"
            run_steal_split "${split_id}" "${gpu_id}" "${slot_id}" "steal"
            if all_task_chunks_done "${output_root}"; then
              echo "[INFO][${task_label}] steal slot=${slot_id} all chunks complete"
              break 2
            fi
          done < <(emit_steal_split_ids)

          if [[ "${attempted}" == "0" ]]; then
            echo "[INFO][${task_label}] steal slot=${slot_id} no unfinished chunks found"
            break
          fi
          sleep "${CHUNK_STEAL_RETRY_SEC}"
        done
      ) &
      pids+=("$!")
    done

    local failed=0
    for pid in "${pids[@]}"; do
      if ! wait "${pid}"; then
        failed=1
      fi
    done
    if [[ "${failed}" == "1" ]]; then
      echo "[ERROR] ${task_label} dagger collect failed on at least one stealing slot."
      return 1
    fi

    for ((split_id=0; split_id<CHUNKS; split_id++)); do
      local chunk_dir="${output_root}/chunk_$(printf "%03d" "${split_id}")"
      local report_path="${report_subdir}/chunk_$(printf "%03d" "${split_id}").json"
      if [[ -d "${chunk_dir}" ]]; then
        mark_chunk_done_if_complete "${chunk_dir}" || true
        echo "[INFO] checking ${task_label} dagger samples for split_id=${split_id}..."
        run_check "${task_name}" "${chunk_dir}" "${report_path}" ""
        report_paths+=("${report_path}")
      fi
    done

    aggregate_reports "${task_name}" "${summary_report}" "${report_paths[@]}"
    maybe_merge_task "${task_name}" "${output_root}"
    return 0
  fi

  local -a pids=()
  local -a report_paths=()
  for ((local_rank=0; local_rank<LOCAL_CHUNKS; local_rank++)); do
    local split_id="${LOCAL_SPLIT_IDS[${local_rank}]}"
    local gpu_id="${GPU_IDS[${local_rank}]}"
    local chunk_dir="${output_root}/chunk_$(printf "%03d" "${split_id}")"
    local chunk_log="${chunk_dir}/runner.log"
    mkdir -p "${chunk_dir}"
    rm -f "${chunk_dir}/chunk.done"

    local -a cmd=(
      "${PYTHON_BIN}" -m "${cli_module}"
      --model-path "${MODEL_PATH}"
      --output-dir "${chunk_dir}"
      --habitat-config-path "${CONFIG_PATH}"
      --dataset-data-path "${dataset_path}"
      --occupancy-root "${occupancy_root}"
      --prompt-type "${PROMPT_TYPE}"
      --astar-margins "${ASTAR_MARGINS}"
      --multifloor-height-threshold "${MULTIFLOOR_HEIGHT_THRESHOLD}"
      --split-num "${CHUNKS}"
      --split-id "${split_id}"
      --start-idx "${START_IDX}"
      --end-idx "${end_idx}"
      --mode collect
    )
    cmd+=(--dagger)
    if [[ -n "${episode_keys_json}" ]]; then
      cmd+=(--episode-keys-json "${episode_keys_json}")
    fi
    if [[ "${RESUME}" == "true" ]]; then
      cmd+=(--resume)
    fi
    if [[ "${GT_STRICT_COVERAGE}" == "true" ]]; then
      cmd+=(--gt-strict-coverage)
    fi
    if [[ "${allow_sliding}" == "true" ]]; then
      cmd+=(--allow-sliding)
    elif [[ "${allow_sliding}" == "false" ]]; then
      cmd+=(--no-allow-sliding)
    fi
    if [[ "${SKIP_MULTIFLOOR}" == "true" ]]; then
      cmd+=(--skip-multifloor)
    fi

    if [[ "${CHUNK_PARALLELISM}" -gt 1 ]]; then
      rm -rf "${chunk_dir}/.queue/claims"
      mkdir -p "${chunk_dir}/.queue/claims" "${chunk_dir}/.queue/done"
      (
        set -euo pipefail
        echo "[INFO][${task_label}] launch split_id=${split_id} gpu=${gpu_id} chunk_dir=${chunk_dir} chunk_parallelism=${CHUNK_PARALLELISM} worker_auto_restart=${WORKER_AUTO_RESTART} worker_restart_delay_sec=${WORKER_RESTART_DELAY_SEC} worker_max_restarts=${WORKER_MAX_RESTARTS}" | tee -a "${chunk_log}"
        worker_pids=()
        for ((queue_worker_id=0; queue_worker_id<CHUNK_PARALLELISM; queue_worker_id++)); do
          worker_log="${chunk_dir}/runner_worker_$(printf "%02d" "${queue_worker_id}").log"
          (
            set -uo pipefail
            restart_count=0
            attempt=1
            while true; do
              echo "[INFO][${task_label}] launch split_id=${split_id} worker=${queue_worker_id}/${CHUNK_PARALLELISM} gpu=${gpu_id} bashpid=${BASHPID} attempt=${attempt} restart_count=${restart_count}" | tee -a "${worker_log}"
              set +e
              CUDA_VISIBLE_DEVICES="${gpu_id}" "${cmd[@]}" --queue-worker-id "${queue_worker_id}" --queue-worker-count "${CHUNK_PARALLELISM}" 2>&1 | tee -a "${worker_log}"
              worker_status="${PIPESTATUS[0]}"
              set -e
              echo "[INFO][${task_label}] worker_exit split_id=${split_id} worker=${queue_worker_id} status=${worker_status} attempt=${attempt} restart_count=${restart_count}" | tee -a "${worker_log}"
              if [[ "${worker_status}" == "0" ]]; then
                echo "[INFO][${task_label}] completed split_id=${split_id} worker=${queue_worker_id}" | tee -a "${worker_log}"
                break
              fi
              if [[ "${WORKER_AUTO_RESTART}" != "true" ]]; then
                exit "${worker_status}"
              fi
              restart_count=$((restart_count + 1))
              if [[ "${WORKER_MAX_RESTARTS}" -gt 0 && "${restart_count}" -gt "${WORKER_MAX_RESTARTS}" ]]; then
                echo "[ERROR][${task_label}] worker_restart_exhausted split_id=${split_id} worker=${queue_worker_id} status=${worker_status} restart_count=${restart_count}" | tee -a "${worker_log}"
                exit "${worker_status}"
              fi
              echo "[WARN][${task_label}] worker_restart split_id=${split_id} worker=${queue_worker_id} status=${worker_status} restart_count=${restart_count} sleep_sec=${WORKER_RESTART_DELAY_SEC}" | tee -a "${worker_log}"
              sleep "${WORKER_RESTART_DELAY_SEC}"
              attempt=$((attempt + 1))
            done
          ) &
          worker_pids+=("$!")
        done
        worker_failed=0
        for worker_pid in "${worker_pids[@]}"; do
          if ! wait "${worker_pid}"; then
            worker_failed=1
          fi
        done
        if [[ "${worker_failed}" == "1" ]]; then
          echo "[ERROR][${task_label}] split_id=${split_id} failed in at least one queue worker" | tee -a "${chunk_log}"
          exit 1
        fi
        touch "${chunk_dir}/chunk.done"
        echo "[INFO][${task_label}] completed split_id=${split_id}" | tee -a "${chunk_log}"
      ) &
      pids+=("$!")
    else
      (
        set -euo pipefail
        echo "[INFO][${task_label}] launch split_id=${split_id} gpu=${gpu_id} chunk_dir=${chunk_dir} worker_auto_restart=${WORKER_AUTO_RESTART} worker_restart_delay_sec=${WORKER_RESTART_DELAY_SEC} worker_max_restarts=${WORKER_MAX_RESTARTS}" | tee -a "${chunk_log}"
        restart_count=0
        attempt=1
        while true; do
          echo "[INFO][${task_label}] single_worker_launch split_id=${split_id} gpu=${gpu_id} attempt=${attempt} restart_count=${restart_count}" | tee -a "${chunk_log}"
          set +e
          CUDA_VISIBLE_DEVICES="${gpu_id}" "${cmd[@]}" 2>&1 | tee -a "${chunk_log}"
          worker_status="${PIPESTATUS[0]}"
          set -e
          echo "[INFO][${task_label}] single_worker_exit split_id=${split_id} status=${worker_status} attempt=${attempt} restart_count=${restart_count}" | tee -a "${chunk_log}"
          if [[ "${worker_status}" == "0" ]]; then
            break
          fi
          if [[ "${WORKER_AUTO_RESTART}" != "true" ]]; then
            exit "${worker_status}"
          fi
          restart_count=$((restart_count + 1))
          if [[ "${WORKER_MAX_RESTARTS}" -gt 0 && "${restart_count}" -gt "${WORKER_MAX_RESTARTS}" ]]; then
            echo "[ERROR][${task_label}] single_worker_restart_exhausted split_id=${split_id} status=${worker_status} restart_count=${restart_count}" | tee -a "${chunk_log}"
            exit "${worker_status}"
          fi
          echo "[WARN][${task_label}] single_worker_restart split_id=${split_id} status=${worker_status} restart_count=${restart_count} sleep_sec=${WORKER_RESTART_DELAY_SEC}" | tee -a "${chunk_log}"
          sleep "${WORKER_RESTART_DELAY_SEC}"
          attempt=$((attempt + 1))
        done
        touch "${chunk_dir}/chunk.done"
        echo "[INFO][${task_label}] completed split_id=${split_id}" | tee -a "${chunk_log}"
      ) &
      pids+=("$!")
    fi
  done

  local failed=0
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      failed=1
    fi
  done
  if [[ "${failed}" == "1" ]]; then
    echo "[ERROR] ${task_label} dagger collect failed on at least one local chunk."
    return 1
  fi

  for split_id in "${LOCAL_SPLIT_IDS[@]}"; do
    local chunk_dir="${output_root}/chunk_$(printf "%03d" "${split_id}")"
    local report_path="${report_subdir}/chunk_$(printf "%03d" "${split_id}").json"
    local expect_arg=""
    if [[ "${FULL}" != "true" && "${CHUNKS}" == "1" && -z "${episode_keys_json}" ]]; then
      expect_arg="${expected_count}"
    fi
    echo "[INFO] checking ${task_label} dagger samples for split_id=${split_id}..."
    run_check "${task_name}" "${chunk_dir}" "${report_path}" "${expect_arg}"
    report_paths+=("${report_path}")
  done

  aggregate_reports "${task_name}" "${summary_report}" "${report_paths[@]}"
  maybe_merge_task "${task_name}" "${output_root}"
}

trap 'echo "Terminating..."; kill 0' SIGINT SIGTERM

if [[ "${RUN_CE}" == "true" ]]; then
  launch_task \
    "ce" \
    "vlnce" \
    "HabitatTools.cli.eval_ce_aligned" \
    "${CE_DATA_PATH}" \
    "${CE_EPISODE_KEYS_JSON}" \
    "${CE_OCC_ROOT}" \
    "${CE_ALLOW_SLIDING}" \
    "${CE_OUT}" \
    "${CE_REPORT}" \
    "${CE_COUNT}"
fi

CE_STATUS="skip"
if [[ -f "${CE_REPORT}" ]]; then
  CE_STATUS="$("${PYTHON_BIN}" - <<PY
import json
from pathlib import Path
print(json.loads(Path("${CE_REPORT}").read_text(encoding="utf-8")).get("status", "unknown"))
PY
)"
fi

echo "[INFO] CE status: ${CE_STATUS}"
echo "[INFO] done: ${OUTPUT_ROOT}"
