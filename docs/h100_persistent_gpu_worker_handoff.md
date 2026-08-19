# H100 Persistent GPU Worker Handoff

This document captures the next implementation direction for MassGen rendering
when the context window is gone. It complements
`docs/h100_sustained_gpu_utilization_plan.md` and should be read before touching
the H100 renderer path.

## Goal

Keep H100 GPUs busy by avoiding repeated scene/avatar loads and by decoupling
GPU rendering from CPU encode and disk writes.

Target pipeline:

```text
global planner / progress ledger
  -> per-GPU scheduler
  -> per-GPU persistent renderer or resource manager
  -> small VRAM output-buffer pool
  -> pinned host-memory frame queue
  -> CPU encode/write workers
  -> completed chunk ledger
```

The current process-per-manifest runner is useful for correctness and smoke
tests, but it spends too much time in Python import, scene load, actor sequence
load, renderer setup, and process lifetime overhead.

## Important Constraint: CUDA IPC Is Not Object Sharing

CUDA IPC can share device memory allocations between processes on the same
machine/GPU. It does not share Python objects, renderer classes, gsplat state,
file handles, CUDA streams, or lifecycle ownership.

If CUDA IPC is used, the shared resources must be represented as explicit device
allocations with metadata:

- scene Gaussian tensors: xyz, opacity, scale, rotation, SH/color;
- avatar/object/robot tensors or frame tensors;
- optional acceleration or per-resource metadata tensors;
- CUDA IPC handles and readiness events;
- reference counts and lease state held by an owner process.

The process that owns an allocation must keep it alive until every worker lease
and every outstanding CUDA event has completed. Workers opening IPC handles must
not outlive the owner allocation.

## Recommended Rollout

### Phase A: Single Persistent Process Per GPU

Implement this before CUDA IPC.

One long-lived process per physical GPU should own:

- CUDA context;
- loaded scene cache;
- loaded avatar/object/robot cache;
- render work queue;
- CUDA streams;
- small output buffer pool;
- async device-to-host copies into pinned RAM;
- metrics emission.

CPU encode/write workers should be separate processes or threads consuming host
RAM chunks. The GPU process should only block when the bounded host queue is
full or when a required resource cannot fit.

Why first:

- validates that removing repeated resource load fixes the dominant bottleneck;
- avoids CUDA IPC lifetime complexity;
- avoids multi-process CUDA context scheduling questions;
- gives a clean baseline before MPS/IPC experiments.

### Phase B: CUDA IPC Resource Manager

Add only if Phase A cannot keep H100 utilization high enough.

Per physical GPU:

- one resource-manager process owns loaded scene/avatar/object allocations;
- multiple GPU render worker processes open IPC handles and render chunks;
- NVIDIA MPS should be benchmarked on/off because multi-process CUDA work can
  serialize or context-switch without it;
- the resource manager owns all eviction and handle lifetime.

This phase should be treated as an optimization experiment, not the first
production path.

## Process Roles

### Global Planner / Progress Manager

Responsibilities:

- read planned missions/render manifests;
- rank and chunk renderable work;
- maintain durable progress ledger;
- submit work in scene/resource-aware order;
- avoid dispatching work whose resource set cannot fit the target GPU budget;
- support stop/drain/restart by chunk boundary.

Ordering inputs:

- scene id and Gaussian model path;
- human avatar ids and action sequences;
- robot/object assets;
- frame count;
- expected visible humans/robots;
- expected stop/turn/straight action value for downstream balancing;
- output chunk id and history-frame requirements.

### Per-GPU Scheduler

Responsibilities:

- hold one GPU's pending queue;
- keep scene-local chunks together;
- prefetch soon-needed resources to RAM;
- ask the GPU process/resource manager to load or evict VRAM resources;
- track live chunks, leases, retries, and failures.

### GPU Resource Owner

Phase A: this is the persistent renderer process.

Phase B: this is the CUDA IPC resource-manager process.

Responsibilities:

- load scene/avatar/object resources once;
- expose resource handles to render work;
- maintain VRAM capacity accounting;
- reject or trigger eviction for work that cannot fit;
- keep reference counts for active work;
- free only resources with no leases and no pending events.

### GPU Render Worker

Phase A: this is an internal worker/loop/stream inside the persistent renderer
process.

Phase B: this may be a separate process opening CUDA IPC handles.

Responsibilities:

- acquire resource leases;
- render frames or chunks;
- copy outputs into host memory;
- release resource leases after CUDA work is done;
- report per-stage timing.

### CPU Encode/Write Workers

Responsibilities:

- consume pinned or regular host-memory chunks;
- encode MP4 with CPU ffmpeg/libx264 on H100 hosts;
- write camera/actor/depth metadata;
- validate output existence and frame count;
- mark chunk complete in the durable ledger.

## Memory Policy

VRAM should hold long-lived resources and a small output-buffer pool:

- current scene Gaussian tensors;
- hot avatars/robots/objects;
- active per-frame actor tensors;
- reusable render output buffers;
- temporary tensors for current render kernels.

VRAM should not hold a deep queue of finished frames. Finished frames should be
copied to pinned host memory quickly and released back to the output buffer
pool.

Host RAM should hold:

- staged resource files or decoded CPU-side resource structures;
- queued frame chunks waiting for encode;
- progress metadata and metrics;
- enough lookahead resources to avoid reading from storage at render time.

Pinned host memory should be bounded. Large pinned allocations can hurt the
host; use a smaller DMA ring and normal RAM for larger staging.

## Resource Lifetime Rules

Every render chunk needs a resource lease set:

```text
scene:<scene_id>/<gaussian_model>
human_avatar:<human_id or action_sequence>
robot_asset:<robot_glb>/<urdf>
object_asset:<asset_id>
output_buffer:<buffer_id>
```

Rules:

- a resource cannot be evicted while lease count > 0;
- an output buffer cannot be reused until its D2H copy event completes;
- a chunk is not complete until CPU encode/write verifies output files;
- if a worker crashes, its leases become suspect and the owner must either
  reclaim them through process death detection or restart the GPU owner;
- if the resource owner crashes, every IPC handle is invalid and workers must
  restart.

## Backpressure

The GPU process should continue rendering while CPU encode/write drains. It
should pause only when:

- host output queue is full;
- pinned buffer pool is exhausted;
- required resources are not loaded and cannot be admitted;
- VRAM fragmentation/OOM requires eviction/restart;
- the global planner requests a drain.

Metrics must include:

- GPU queue wait time;
- host output queue wait time;
- pinned buffer wait time;
- D2H copy time;
- CPU encode time;
- disk write time;
- resource load/reuse/evict counts;
- cache hit rate by resource kind.

## Preemption Model

The platform may need to release GPUs by killing the VM/workers. Design for
hard preemption:

- chunks should be small enough to rerun cheaply;
- outputs are written into temporary directories;
- a chunk is complete only after `TASK_DONE.json` or equivalent durable marker;
- startup scans existing markers and resumes incomplete work;
- stopping should request drain when possible, but hard kill must be safe.

## Implementation Status

Done:

- lifecycle/setup timing exists in renderer metrics;
- `run_render_smoketest_benchmark.py` supports scene ordering,
  preemptible-output temp dirs, and resume markers;
- 5880 whole-scene test showed process launches can be reduced to one, but GPU
  utilization stayed low because actor sequence loading still dominated.

Started in this session:

- `navdp_datagen.massgen.persistent_scheduler` defines a planner-facing work
  item/resource/chunk/cache abstraction;
- `scripts/massgen/plan_persistent_h100_schedule.py` dry-runs persistent-worker
  scheduling from an existing render plan JSON;
- tests cover resource extraction, scene chunking, GPU assignment, and cache
  eviction/refcount behavior.

Next implementation steps:

1. Teach the persistent scheduler to read the formal smoke package directly,
   not only render plan JSON.
2. Add a single-process persistent renderer prototype that can consume one
   `PersistentGpuSchedule` for camera-only or actor-plan jobs.
3. Move actor sequence loading into a process-local cache keyed by actor/action
   resource keys.
4. Add bounded async host output queue and CPU encode workers.
5. Benchmark Phase A on 5880 with the same 100-mission scene used in
   `out/scene_order_5880/9d8867b/chingmu3_0016`.
6. Only after Phase A, test CUDA IPC + MPS as a separate comparison.
