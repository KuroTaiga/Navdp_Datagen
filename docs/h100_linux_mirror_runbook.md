# H100 Linux Mirror Runbook

This runbook is for creating a Linux/H100 mirror of the MassGen renderer after
the persistent H100 pipeline has been validated on 5880.

## Goal

Keep the H100 platform checkout isolated from 5880-specific test state while
using the same `massgen` branch implementation:

- persistent scene/resource-aware planning;
- multiple logical worker lanes per physical H100;
- CPU `libx264` video encoding;
- taskset CPU affinity and per-worker thread caps;
- preemptible chunk outputs and resume markers;
- 10 Hz-ish GPU/VRAM sampling and full-run lifecycle reports.

## Suggested Layout

```text
/team/telenav/
  code/Navdp_Datagen/             # git mirror, branch massgen
  massgen_packages/<package>/     # copied planner/render package
  h100_results/<run-id>/          # run output root
  human_avatars/...               # human avatar source, if not already elsewhere
```

On the current H100 platform, only `/team` and `/private` are persistent. Keep
code under `/team/telenav/code`; do not rely on `/root`, `/dev/shm`, or the
container root filesystem surviving a restart. Do not write large results into
the git checkout.

## Mirror Setup

```sh
cd /team/telenav/code
git clone <repo-url> Navdp_Datagen
cd Navdp_Datagen
git switch massgen
git pull --ff-only origin massgen
```

Verify the environment:

```sh
/path/to/cuda-env/bin/python -m py_compile \
  render_label_paths_telesim.py \
  scripts/massgen/run_family_rollout_h100.py \
  scripts/massgen/plan_persistent_h100_schedule.py \
  scripts/massgen/run_persistent_h100_schedule.py \
  scripts/massgen/report_persistent_h100_schedule_run.py

/path/to/cuda-env/bin/python - <<'PY'
import torch
print("cuda", torch.cuda.is_available(), "count", torch.cuda.device_count())
PY

nvidia-smi
```

Also verify the package references existing scene and avatar roots. The
renderer does not copy scene Gaussian PLYs or human avatar PLY sequences into
the package; it expects referenced paths to exist on the H100 host.

## Helper Script

This repo includes a Linux mirror helper:

```sh
MIRROR_ROOT=/team/telenav/code/Navdp_Datagen \
REPO_URL=https://github.com/KuroTaiga/Navdp_Datagen.git \
BRANCH=massgen \
scripts/massgen/setup_h100_linux_mirror.sh
```

Optional package copy:

```sh
MIRROR_ROOT=/team/telenav/code/Navdp_Datagen \
PACKAGE_SRC=/path/to/source/package \
PACKAGE_DST=/team/telenav/massgen_packages/<package> \
scripts/massgen/setup_h100_linux_mirror.sh
```

## Smoke Command

Start with a capped run before a full natural-length run:

```sh
cd /team/telenav/code/Navdp_Datagen
H100_PYTHON=/team/telenav/code/conda_envs/navdp_cuda121/bin/python

"${H100_PYTHON}" scripts/massgen/run_family_rollout_h100.py \
  --package-root /team/telenav/massgen_packages/<package> \
  --results-root /team/telenav/h100_results/<run-id> \
  --python-bin "${H100_PYTHON}" \
  --gpu-devices 0,1,2,3 \
  --cpu-cores 120 \
  --jobs-per-gpu 4 \
  --video-backend cpu \
  --renders-per-family-source-scene 50 \
  --minimal-frames 16 \
  --command-attempts 3 \
  --clean
```

`run_family_rollout_h100.py` derives `PYOPENGL_PLATFORM=egl` and prefixes
`<python-env>/lib` on `LD_LIBRARY_PATH` from `--python-bin`, so the persistent
conda env can provide `libGL`, `libEGL`, and `libGLU` without relying on root
filesystem apt packages. If invoking render scripts directly, export those
values first:

```sh
export PYOPENGL_PLATFORM=egl
export LD_LIBRARY_PATH=/team/telenav/code/conda_envs/navdp_cuda121/lib:${LD_LIBRARY_PATH:-}
```

Outputs to inspect first:

```text
<results-root>/run_persistent/benchmark_summary.json
<results-root>/report_persistent/REPORT.md
<results-root>/report_persistent/assets/graphs/full_run_stage_overlay.png
<results-root>/report_persistent/assets/tables/full_run_worker_lanes.csv
<results-root>/mp4_count.txt
```

## Container Option

The renderer-side H100 container is separate from the Pathplanner CPU MassGen
container. Build it from this repo mirror:

```sh
cd /team/telenav/code/Navdp_Datagen
IMAGE_TAG=navdp-datagen-h100:massgen scripts/massgen/build_h100_clean_mirror_image.sh
```

The clean builder exports only git-tracked superproject files into a temporary
Docker context, checks that the checkout is clean, targets `linux/amd64`, and
defaults to `--pull --no-cache`. It does not require renderer submodules; the
image installs the render dependency set from
`release/navdp_path_renderer/requirements.txt`, including `gsplat`, with pip.
For an iterative non-release build from the live checkout, use
`scripts/massgen/build_h100_container.sh` instead.

If the image must be moved to another Linux/H100 host without rebuilding:

```sh
SAVE_IMAGE_TAR=/team/telenav/navdp-datagen-h100_massgen.tar \
  IMAGE_TAG=navdp-datagen-h100:massgen \
  scripts/massgen/build_h100_clean_mirror_image.sh
```

Run the capped smoke in the container:

```sh
PACKAGE_ROOT=/team/telenav/massgen_packages/<package> \
RESULTS_ROOT=/team/telenav/h100_results/<run-id> \
GPU_DEVICES=0,1,2,3 \
CPU_CORES=120 \
JOBS_PER_GPU=4 \
MINIMAL_FRAMES=16 \
RENDERS_PER_FAMILY_SOURCE_SCENE=50 \
EXTRA_H100_ARGS="--clean" \
scripts/massgen/run_h100_container.sh run
```

`run_h100_container.sh` mounts common data roots at the same absolute path
inside the container (`/mnt/DATA`, `/mnt/DATA1`, `/private_lxh`,
`/team/telenav`) when those paths exist. If the package references another
absolute root, add an explicit mount or adjust the script before running.

## Resume And Preemption

The persistent runner writes chunks into temporary directories and promotes
them only after success. If the VM is killed or GPUs must be released, rerun the
same command without `--clean`; `--resume` is enabled by default.

Use `--clean` only when intentionally discarding previous partial output.

## Full Run

For full natural-length generation, omit `--minimal-frames` and remove smoke
caps such as `--renders-per-family-source-scene`. Keep `--jobs-per-gpu 4` for
the first full run, then tune upward only if:

- GPU average is below 80%;
- host CPU/RAM are not saturated;
- CPU encode/write stages are not building a backlog;
- peak VRAM has enough margin for the selected scene/avatar mix.
