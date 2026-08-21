# syntax=docker/dockerfile:1.6

ARG CUDA_BASE=nvidia/cuda:12.1.1-devel-ubuntu22.04
FROM ${CUDA_BASE}

ARG CONDA_ENV=cuda121
ARG MINIFORGE_VERSION=24.11.3-0
ARG INSTALL_LOCAL_EXTENSIONS=1

ENV DEBIAN_FRONTEND=noninteractive \
    CONDA_DIR=/opt/conda \
    CONDA_ENV=${CONDA_ENV} \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
    GAUSSIAN_RENDER_BACKEND=gsplat \
    PYOPENGL_PLATFORM=egl \
    MPLCONFIGDIR=/tmp/navdp_mplconfig \
    XDG_CACHE_HOME=/tmp/navdp_xdg_cache \
    FC_CACHEDIR=/tmp/navdp_fontconfig

SHELL ["/bin/bash", "-lc"]

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    ffmpeg \
    git \
    git-lfs \
    libegl1 \
    libgl1 \
    libglib2.0-0 \
    libglvnd0 \
    libgomp1 \
    libosmesa6 \
    libsm6 \
    libxext6 \
    libxrender1 \
    ninja-build \
    openssh-client \
    pkg-config \
    rsync \
    tini \
    util-linux \
    wget \
  && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL \
      "https://github.com/conda-forge/miniforge/releases/download/${MINIFORGE_VERSION}/Miniforge3-Linux-x86_64.sh" \
      -o /tmp/miniforge.sh \
  && bash /tmp/miniforge.sh -b -p "${CONDA_DIR}" \
  && rm /tmp/miniforge.sh \
  && "${CONDA_DIR}/bin/conda" config --system --set auto_update_conda false \
  && "${CONDA_DIR}/bin/conda" config --system --set channel_priority flexible

ENV PATH=${CONDA_DIR}/envs/${CONDA_ENV}/bin:${CONDA_DIR}/bin:${PATH}

WORKDIR /workspace/Navdp_Datagen

COPY environment.yml /tmp/environment.yml
RUN conda env create -f /tmp/environment.yml \
  && conda clean -afy \
  && mkdir -p "${MPLCONFIGDIR}" "${XDG_CACHE_HOME}" "${FC_CACHEDIR}"

COPY . /workspace/Navdp_Datagen

RUN if [[ "${INSTALL_LOCAL_EXTENSIONS}" == "1" ]]; then \
      if [[ -f submodules/gsplat/pyproject.toml ]]; then \
        python -m pip install --no-build-isolation -e submodules/gsplat; \
      fi; \
      if [[ -f submodules/diff-gaussian-rasterization/setup.py ]]; then \
        python -m pip install --no-build-isolation -e submodules/diff-gaussian-rasterization; \
      fi; \
      if [[ -f submodules/simple-knn/setup.py ]]; then \
        python -m pip install --no-build-isolation -e submodules/simple-knn; \
      fi; \
      if [[ -f submodules/fused-ssim/setup.py ]]; then \
        python -m pip install --no-build-isolation -e submodules/fused-ssim; \
      fi; \
    fi \
  && python -m py_compile \
      render_label_paths_telesim.py \
      scripts/massgen/run_family_rollout_h100.py \
      scripts/massgen/plan_persistent_h100_schedule.py \
      scripts/massgen/run_persistent_h100_schedule.py \
      scripts/massgen/report_persistent_h100_schedule_run.py

RUN chmod +x /workspace/Navdp_Datagen/containers/h100/entrypoint.sh

ENTRYPOINT ["/usr/bin/tini", "--", "/workspace/Navdp_Datagen/containers/h100/entrypoint.sh"]
CMD ["preflight"]
