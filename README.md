# GN0-VLN-CE

## Overview

GN0-VLN-CE is the CE evaluation branch of the GN0 project. It is used to run
GN0/BAE checkpoints in Habitat VLN-CE with Habitat-Sim and MP3D scene assets.

This branch focuses on the CE evaluation and CE-aligned data-collection
workflow:

- run GN0/BAE checkpoint evaluation on R2R VLN-CE splits;
- collect Habitat-aligned CE DAgger trajectories;
- analyze CE metrics, per-episode logs, and chunked runs.

The main runtime pieces are:

- `bae/`: BAE model inference, prompts, and parsing utilities.
- `bae_agent_eval.py`: the formal CE evaluation agent.
- `bae_agent_dagger.py`: the CE DAgger/data-collection agent.
- `habitat-tools/`: the Habitat CE environment, evaluator, metrics, and adapter
  layer.
- `tools/`: data layout checks, occupancy map building, metric analysis, and
  run monitors.
- `eval_ce.sh` and `dagger_ce.sh`: top-level launchers for evaluation and DAgger
  collection.

This repo expects CE trajectory files as Habitat VLNCE episode JSON files, MP3D
scene folders under `data/scene_datasets/mp3d`, and precomputed CE occupancy
maps under `data/scene_datasets/mp3d_ce_occ` for the DAgger correction path.

## Installation

### 1. Create the conda environment

```bash
conda create -n gn0_vln_ce python=3.9
conda activate gn0_vln_ce

# Reinstall a pip version that supports Python 3.9
conda install -y "pip<26" wheel setuptools
```

### 2. Clone and build Habitat-Sim / Habitat-Lab in `thirdparty`

```bash
cd /path/to/GN0-VLN-CE
mkdir -p thirdparty
cd thirdparty

git clone --branch v0.1.7 https://github.com/facebookresearch/habitat-sim.git
cd habitat-sim

pip install -r requirements.txt

sudo apt-get update || true
sudo apt-get install -y --no-install-recommends \
    libjpeg-dev libglm-dev libgl1-mesa-glx libegl1-mesa-dev \
    mesa-utils xorg-dev freeglut3-dev

python setup.py install --headless --with-cuda

cd ..
git clone --branch v0.1.7 https://github.com/facebookresearch/habitat-lab.git
cd habitat-lab
pip install -e .
```

### 3. Install PyTorch and Transformers

```bash
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
conda install -y -c conda-forge "pandas<3" av pyarrow orjson ffmpeg
pip install --ignore-requires-python llamafactory==0.9.4
```

### 4. Install `bae`

```bash
cd /path/to/GN0-VLN-CE
pip install -e ./bae
```

### 5. Download the GN-BAE VLN-CE checkpoint

The default launchers expect the model checkpoint at
`models/gn-bae-vln-ce`. Download the CE checkpoint from
[TeleEmbodied/GN-BAE-VLN-CE](https://huggingface.co/TeleEmbodied/GN-BAE-VLN-CE):

```bash
cd /path/to/GN0-VLN-CE
pip install -U huggingface_hub hf_xet
huggingface-cli download TeleEmbodied/GN-BAE-VLN-CE \
  --local-dir models/gn-bae-vln-ce
```

The model repository is about 17.6 GB and contains BF16 safetensors weights. If
you store the checkpoint elsewhere, set `MODEL_PATH` when launching eval or
DAgger:

```bash
MODEL_PATH=/path/to/models/gn-bae-vln-ce bash eval_ce.sh
MODEL_PATH=/path/to/models/gn-bae-vln-ce bash dagger_ce.sh
```

Before starting a run, make sure the launcher points to the checkpoint you want
to evaluate. The top-level `eval_ce.sh` and `dagger_ce.sh` call the Habitat
launchers under `habitat-tools/scripts/`, whose default model path is
`models/gn-bae-vln-ce`. If you are not using that location, either pass
`MODEL_PATH` as shown above or update `DEFAULT_MODEL_PATH` in:

```text
habitat-tools/scripts/eval_habitat_bae_vlnce_aligned.sh
habitat-tools/scripts/run_habitat_aligned_dagger_data.sh
```

## Data Layout

GN0-VLN-CE follows the same CE data split idea as the InternNav dataset
preparation guide: VLNCE episode JSON files are kept separately from MP3D
scene assets. In this repo, use the following layout:

### Download CE data

Follow the InternNav dataset preparation guide for the source downloads:

- CE trajectory / episode files: [InternData-N1](https://huggingface.co/datasets/InternRobotics/InternData-N1), download the `vln_ce` subset.
- MP3D CE scene assets: [Scene-N1](https://huggingface.co/datasets/InternRobotics/Scene-N1), download the `mp3d_ce` subset.

Both Hugging Face datasets may require you to log in and accept the dataset
license before downloading. A selective `git-lfs` download is usually enough
for this repo:

```bash
cd /path/to/downloads
git lfs install

# Download VLNCE episode definitions.
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/datasets/InternRobotics/InternData-N1
cd InternData-N1
git lfs pull --include="vln_ce/raw_data/r2r/**"
mkdir -p /path/to/GN0-VLN-CE/data/datasets/R2R_VLNCE_v1-3_preprocessed
rsync -a vln_ce/raw_data/r2r/ \
  /path/to/GN0-VLN-CE/data/datasets/R2R_VLNCE_v1-3_preprocessed/

# Download MP3D CE scene assets.
cd /path/to/downloads
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/datasets/InternRobotics/Scene-N1
cd Scene-N1
git lfs pull --include="scene_data/mp3d_ce/**"
mkdir -p /path/to/GN0-VLN-CE/data/scene_datasets/mp3d
```

After downloading Scene-N1, copy the scene-id folders into
`data/scene_datasets/mp3d`. Depending on the downloaded package layout, the
source may be either `scene_data/mp3d_ce/` or `scene_data/mp3d_ce/mp3d/`:

```bash
# If scene folders are directly under scene_data/mp3d_ce:
rsync -a scene_data/mp3d_ce/ /path/to/GN0-VLN-CE/data/scene_datasets/mp3d/

# If there is an extra mp3d level:
rsync -a scene_data/mp3d_ce/mp3d/ /path/to/GN0-VLN-CE/data/scene_datasets/mp3d/
```

Then build the CE occupancy maps used by the DAgger correction path:

```bash
cd /path/to/GN0-VLN-CE
bash tools/build_occupancy_ce.sh
```

Verify that the CE data layout matches what the launchers expect:

```bash
cd /path/to/GN0-VLN-CE
bash tools/verify_data_links.sh
```

The verifier checks the R2R VLN-CE split files, MP3D scene folders,
`.glb`/`.navmesh` scene assets, CE occupancy maps, and scene ids referenced by
the episode files. It writes a manifest to `data_link_manifest.txt` and exits
with a non-zero status if required files or directories are missing.

By default, the episode scene cross-check uses `val_unseen`, matching the
default CE evaluation split and the default `tools/build_occupancy_ce.sh`
occupancy build. For train DAgger collection or a full data check, pass the
splits explicitly:

```bash
VERIFY_SPLITS=train,val_seen,val_unseen bash tools/verify_data_links.sh
```

If your data is symlinked or stored outside the repo, override the checked
paths:

```bash
DATASET_ROOT=/path/to/R2R_VLNCE_v1-3_preprocessed \
MP3D_ROOT=/path/to/mp3d \
OCCUPANCY_ROOT=/path/to/mp3d_ce_occ \
VERIFY_SPLITS=val_unseen \
bash tools/verify_data_links.sh
```

The final layout should be:

```text
GN0-VLN-CE
├── data
│   ├── datasets
│   │   └── R2R_VLNCE_v1-3_preprocessed
│   │       ├── train
│   │       │   └── train.json.gz
│   │       ├── val_seen
│   │       │   └── val_seen.json.gz
│   │       └── val_unseen
│   │           └── val_unseen.json.gz
│   └── scene_datasets
│       ├── mp3d
│       │   ├── 17DRP5sb8fy
│       │   │   ├── 17DRP5sb8fy.glb
│       │   │   ├── 17DRP5sb8fy.navmesh
│       │   │   └── ...
│       │   ├── 1LXtFkjw3qL
│       │   └── ...
│       └── mp3d_ce_occ
│           ├── 17DRP5sb8fy
│           │   ├── occupancy.json
│           │   └── occupancy.png
│           ├── 1LXtFkjw3qL
│           └── ...
└── models
    └── gn-bae-vln-ce
```

The trajectory files under `data/datasets/R2R_VLNCE_v1-3_preprocessed`
are the Habitat VLNCE episode definitions: instructions, start poses, goals,
and split membership. They correspond to the `vln_ce/raw_data/r2r` split files
in the InternNav documentation.

The MP3D scene folders under `data/scene_datasets/mp3d` contain the Habitat
scene assets used by Habitat-Sim. The scene id in each trajectory must match a
folder under this directory.

The `data/scene_datasets/mp3d_ce_occ` directory stores precomputed occupancy
maps used by the BAE DAgger correction/planning path.

The default launchers expect these paths:

```bash
# eval_ce.sh
DATASET_DATA_PATH=data/datasets/R2R_VLNCE_v1-3_preprocessed/val_unseen/val_unseen.json.gz
SCENES_DIR=data/scene_datasets
OCCUPANCY_ROOT=data/scene_datasets/mp3d_ce_occ

# dagger_ce.sh
CE_DATA_PATH=data/datasets/R2R_VLNCE_v1-3_preprocessed/train/train.json.gz
CE_OCC_ROOT=data/scene_datasets/mp3d_ce_occ
```

You can override any of them through environment variables or the corresponding
launcher flags.

### 5. Python interpreter selection

The launcher scripts no longer depend on a hard-coded conda environment name. By default they use the current `python` in your active shell.

If you want to be explicit, set:

```bash
cd /path/to/GN0-VLN-CE
export PYTHON_BIN="$(which python)"
```

For `tools/monitor_dagger_progress.py`, you can also pass:

```bash
python tools/monitor_dagger_progress.py --python-bin "$(which python)" ...
```

## Result Analysis

CE runs already write `eval_result.log` and `result.json` while evaluating. To
recompute and inspect the aggregate metrics from the current per-episode output
format, use:

```bash
cd /path/to/GN0-VLN-CE
python tools/analyze_total_metrics.py --path /path/to/eval_run
```

The script auto-detects both plain CE eval runs:

```text
/path/to/eval_run/merged/progress.jsonl
```

and DAgger CE collection runs:

```text
/path/to/dagger_run/ce_run/merged/progress.jsonl
```

It also falls back to `chunk_*/progress.jsonl` or `log/*.json` if the merged
output is not available. To save the recomputed summary:

```bash
python tools/analyze_total_metrics.py \
  --path /path/to/eval_run \
  --output-json /path/to/eval_run/total_metrics.json
```
