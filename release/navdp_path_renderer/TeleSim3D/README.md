# TeleSim3D
Simulation backbone for 3DGS scenes

## Environment Setup

### Quick Start (Conda)

Use the helper script to create or update the TeleSim3D environment:

```bash
./scripts/setup_env.sh
conda activate telesim3d
python -m pip install -e .
```

The script wraps `conda env create/update` so repeated runs stay in sync with
`environment.yml`. Pass `--update-only` to avoid creating the environment and
`--extras navgen` to pull in the NavGen pipeline dependencies.

### Manual Steps

If you prefer to drive Conda yourself:

```bash
conda env create -f environment.yml
conda activate telesim3d
python -m pip install -e .
```

The specification installs Python 3.10, GPU-enabled PyTorch 2.0.1 (CUDA 11.8),
torchvision 0.15.2, numpy, plyfile, requests, pytest, imageio, and pip. Update
`environment.yml` whenever runtime or testing dependencies change so future
setups remain reproducible.

### Optional Extras

Install the NavGen task-generation dependencies after activating the environment:

```bash
./scripts/setup_env.sh --extras navgen
```

This pulls `opencv-python`, `Pillow`, `timm==0.4.12`, `transformers>=4.25.1`,
`fairscale==0.4.4`, `pycocoevalcap`, `pycocotools`, `openai`, `tqdm`,
`ruamel.yaml`, `scipy`, `matplotlib`, `PyQt5`, and the OpenAI CLIP git package.
`git` and a working compiler toolchain are required—`pycocotools` builds native
extensions and the CLIP package clones from GitHub.

### Dependency Notes & Conflicts

- PyTorch 2.0.1, torchvision 0.15.2, and `pytorch-cuda=11.8` target CUDA 11.8
  and require NVIDIA driver ≥ 520. On CPU-only hosts swap `pytorch-cuda` for
  `cpuonly` (from the `pytorch` channel) or install CPU wheels manually.
- Habitat-Sim must be built against Python 3.10 and a compatible CUDA toolkit.
  Mixing binaries compiled for a different CUDA/Python pair frequently yields
  `ImportError: magnum` or navmesh recompute crashes; rebuild Habitat-Sim after
  upgrading Python, CUDA, or numpy.
- The NavGen extras rely on `fairscale==0.4.4`, which only ships wheels for
  Linux x86_64. Other platforms need a full compiler toolchain to build from
  source, and PyTorch 2.0+ support is limited in older releases.
- `pycocotools`/`pycocoevalcap` require C/C++ build tools (GCC/Clang) and fail
  on Windows without WSL or Visual Studio Build Tools.
- `PyQt5` depends on desktop Qt libraries. Skip the `navgen` extra if you do not
  need the GUI components, or ensure the required system packages are present.
- Installing the CLIP extra needs outbound network access to GitHub.

### GPT-4o Planner Integration

Natural language tasking requires the GPT-4o deployment. Provide credentials via
environment variables (they can live in a local `.env` file):

```bash
export CHAT4O_ENDPOINT="https://<resource>.cognitiveservices.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2025-01-01-preview"
export CHAT4O_API_KEY="<your-azure-openai-key>"
```

Both variables must be defined; the simulator raises an error otherwise.

#### Humanoid Action Sections

- The NL→plan translator emits ordered **action sections** for each avatar. Every
  section includes an `action_type`, the `animation` folder name, `start`/`end`
  locations, and a frame count (30 FPS default). Relocation segments (e.g.
  `Walking`, `Running`) are tagged to use Habitat-Sim's
  `GreedyGeodesicFollower` for path execution.
- Frame counts are estimated from navmesh paths: relocation sections divide the
  computed path length by the action's speed, while gestures/interactions fall
  back to clip durations defined in `action_constants`.

#### Testing Notes

- `pytest tests/test_nl2plan.py` now calls the live Chat4o deployment. Set
  `CHAT4O_ENDPOINT` / `CHAT4O_API_KEY` before running; otherwise the suite skips
  with a helpful message. Use `pytest -s` if you want to view the emitted plans.
- Override the integration tests with `NL2PLAN_TEST_*` env vars (e.g.
  `NL2PLAN_TEST_PROMPT`, `NL2PLAN_TEST_BEV`) or run a one-off translation via
  `python tests/test_nl2plan.py "your prompt" --bev path/to/bev.png --nav path/to/nav.png`.
- Supported animations are catalogued in `tele_sim/plans/action_constants.py`,
  which reflects folders under `assets/Actions`. Extend this module when adding
  new clips so prompt assembly and validation stay in sync.

### Version Compatibility

- `python=3.10`, `pytorch=2.0.1`, and `pytorch-cuda=11.8` match the Gaussian
  Splatting reference build; keep these aligned if you rebuild either project.
- If you maintain a CPU-only toolchain, replace `pytorch-cuda` with
  `cpuonly` (from the `pytorch` channel) to avoid pulling GPU runtimes.
- When reusing an existing environment, verify compatibility with
  `python -c "import torch; import platform; print(platform.python_version(), torch.__version__)"`.

## External Dependencies

TeleSim3D consumes rendering utilities from the upstream Gaussian Splatting
reference implementation.

Additional runtime dependency:

- `habitat-sim` powers navigation mesh queries and path planning. Build it from
  source using the official repository:

  ```bash
  git clone https://github.com/facebookresearch/habitat-sim.git
  cd habitat-sim
  python setup.py install --headless --with-cuda
  ```

  Adjust build flags as needed for your platform (see Habitat-Sim docs).

### NavMesh & BEV Generation Helper

Once Habitat-Sim is available, regenerate avatar-specific navigation assets
(navmesh + BEV raster) using:

```bash
python -m tele_sim.tools.nav_asset_builder \
  --scene-glb /media/dx/DATA/Habitat_test/scene_datasets/<scene-path>.glb \
  --agent-radius 0.4 \
  --output-dir assets/scenes
```

Outputs `<scene>_rXX.navmesh`, `<scene>_bev.png`, and a companion
`<scene>_bev.json` metadata file aligned with the GPT-4o prompt pipeline, plus
`<scene>_scene.json` summarising semantic regions/objects. Max climb defaults to
0.2 m to disable stair climbing. Supply `--dataset-config` if the GLB is not part
of a dataset with an implicit `.scene_dataset_config.json`, `--slice-offset` to
shift the BEV slice relative to the navmesh minimum bound, `--slice-height` to
set the absolute height (deprecated; prefer offsets), and `--disable-sliding` to
turn off Habitat's collision sliding during generation.

During prompt iteration, drop the generated outputs in `tmp_test_results/`.
When a `<scene>_bev.json` or `<scene>_bev.png` exists there, the NL→plan
translator automatically attaches that BEV image to GPT requests so you can
refine prompts without modifying the versioned assets.

### Prompt-to-Plan Pipeline

Use `scripts/prompt_to_plan.py` to automate the full workflow:

```bash
python scripts/prompt_to_plan.py \
  "Create a person in the top showroom, walk the corridor, and return." \
  --scene-glb assets/Scenes/floor_30_base/floor30_rot.glb \
  --output-dir tmp_test_results/floor30_rot_default
```

This helper will:

1. Build (or reuse) navigation assets for the requested avatar footprint (defaults: radius 0.10 m, height 1.70 m, meters-per-pixel 0.10).
2. Invoke the GPT-4o translator with the generated BEV/zone metadata.
3. Augment relocation actions with Habitat shortest-paths, then persist the plan JSON and BEV overlay diagnostics.

Flags mirror the nav-asset builder (`--avatar-radius`, `--slice-height`, `--force-assets`, etc.). Logs land beside the output directory (e.g. `<scene>_YYYYMMDD-HHMMSS_pipeline.log`) and capture the raw Chat4o response plus Habitat warnings for later inspection.

#### Gradio Demo

Launch an interactive demo with Gradio:

```bash
python scripts/gradio_demo.py
```

Select a scene GLB (the default browser lists files under `/media/dx/DATA/TeleSim3D_Assets/Scenes`), tweak avatar/navmesh parameters, and type a prompt. The UI displays the raw GPT-4o response, the generated plan JSON, a BEV+zone overlay, and either the final path overlay or any diagnostic image produced during path planning. Install Gradio (`pip install gradio`) if it is not already available in the environment.

- Use the **Use local .env / environment credentials** toggle to decide whether the app should read `CHAT4O_ENDPOINT` / `CHAT4O_API_KEY` from your `.env`/shell or rely on the custom fields. When the toggle is off, the endpoint and API key inputs become writable so you can paste ad‑hoc credentials without editing local environment files.
- Habitat path overlays now mark the traversal order: each sampled waypoint is highlighted and labelled with its 1-indexed step number so you can inspect turn-by-turn pathing.
- To visualise Gaussian Splatting reconstructions alongside the navmesh BEV,
  install the upstream renderer once:

  ```bash
  git clone https://github.com/graphdeco-inria/gaussian-splatting.git ../gaussian-splatting
  pip install -e ../gaussian-splatting
  ```

- Then pass `--splat-ply /path/to/scene/point_cloud.ply` to
  `tele_sim.tools.nav_asset_builder` to emit an additional
  `<scene>_splats_bev.png` sharing the same grid resolution as the navmesh map.

- If you already have a local checkout, point TeleSim3D at it via
  `GAUSSIAN_SPLATTING_ROOT=/path/to/gaussian-splatting` or install it into the
  environment with `pip install -e /path/to/gaussian-splatting`.
- Otherwise, clone the repository and build it following their instructions:

  ```bash
  git clone https://github.com/graphdeco-inria/gaussian-splatting.git ../gaussian-splatting
  ```

  Then set `GAUSSIAN_SPLATTING_ROOT` (or update renderer config) so TeleSim3D can
  locate its compiled utilities.

## Future Work

- **Multiple avatars:** extend the prompt schema so GPT-4o can produce plans for several people, then validate path augmentation per avatar and colour-code overlays for clarity.
- **Runtime playback:** drive the generated plans through the animation controller to ensure relocation frames and `path_points` move avatars correctly on the navmesh.
- **Expanded actions:** grow `tele_sim/prompts/nav_task/actions.json` with richer gestures/interactions and add regression tests to verify the translator classifies them properly.
