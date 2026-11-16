# vjumpkung ComfyUI Memory Profiler

Records the VRAM and RAM used by the ComfyUI process and per-block model residency for every run and saves the result as JSON and a PNG report. Requires matplotlib (`pip install -r requirements.txt`).

## What is recorded

- After every executed node: process VRAM, aimdo VRAM, torch allocated/reserved/peak, process RAM (RSS), pinned memory and the VRAM/RAM capacity, plus every loaded model with its size, loaded bytes, pinned bytes, host RAM and per-block VRAM.
- At the end of the run: the same snapshot, the memory held by cached node outputs grouped by type (`IMAGE`, `LATENT`, `CONDITIONING`, ...), and optionally every live torch tensor grouped by device and dtype.

The PNG is a matplotlib report with a header, VRAM and RAM charts of the ComfyUI process only, the node list, a residency heatmap per model (columns are nodes in execution order, colour is the fraction of the block in VRAM, grey is not loaded) and the output and tensor tables.

## Usage

Add a **Memory Profiler** node (`utils/memory_profiler`) to the workflow. It has no outputs; its presence enables profiling for that run. Files are written to `output/vjumpkung_memory_profiler/<file_prefix>_<name_prefix>_<file_suffix>_00001_.json` and `.png`. Empty `file_prefix` and `file_suffix` are skipped, and `/` and `\` in the name become `_`. Without a Memory Profiler node (`require_node: false`) only `name_prefix` is used.

Right-click the node for an instant snapshot (JSON, PNG or both) of the current memory state, named from the node's current `file_prefix` and `file_suffix`. Snapshots are refused while a prompt is running.

**Memory Profile to Image** renders a saved JSON profile to an image and writes the PNG next to it. Press `R` to refresh the profile list after new files appear.

## Config

`config.yaml` is created from `config.yaml.example` on first run and re-read at the start of every prompt.

| key | default | meaning |
| --- | --- | --- |
| `name_prefix` | `memory_profile` | file name between the node's `file_prefix` and `file_suffix` |
| `require_node` | `true` | only profile prompts that contain a Memory Profiler node |
| `save_format` | `both` | `json`, `png` or `both` |
| `per_node` | `true` | snapshot after every executed node |
| `block_residency` | `true` | per-block VRAM of loaded models in each snapshot |
| `cached_outputs` | `true` | memory held by node outputs at the end of the run |
| `tensor_scan` | `false` | scan all live torch tensors at the end of the run and in manual snapshots (slow) |

## JSON layout

```
version (2), kind ("run" | "manual"), created, device, aimdo, config
prompt_id, success, seconds                  (runs only)
timeline: [{node_id, class_type, status, seconds, memory, models}]
final: {memory, models, outputs?, tensors?}
memory: {vram_total, process_vram, aimdo_vram, torch_allocated, torch_reserved, torch_peak,
         ram_total, process_rss, pinned}
models: [{id, name, device, dynamic, size, loaded, pinned, host_ram, blocks?: [{name, size, vram}]}]
outputs: {by_type: {type: {device: bytes}}, by_node: [{node_id, class_type, type, bytes}]}
tensors: [{device, dtype, count, bytes}]
```

Values that are unavailable on the current device are `null`.

## Caveats

- Hooks wrap `execution.PromptExecutor.execute` and `execution.execute`.
- With `per_node` the torch peak counter is reset before every node, so `torch_peak` in the timeline is the peak inside that node. In manual snapshots it is the peak since the last profiled node.
- `process_vram` is the aimdo tally plus the free reserve of the cudaMallocAsync pool when aimdo is on, and torch reserved otherwise. It excludes the CUDA context and allocations made outside torch and aimdo, so it is lower than Task Manager or `nvidia-smi`.
- `process_rss` is the process working set (RSS).
- Dynamic VRAM block residency is read from aimdo VBAR pages (32 MiB granularity). Block `vram` is the stored weight size scaled by the resident fraction, so weights that are cast to a smaller dtype on load (for example fp32 weights run as fp16) report more than the real VRAM use. Compare with the model's `loaded` value.
- Cached output sizes cover top-level node outputs only. Tensors shared between outputs are counted once.
- `tensor_scan` walks every Python object and can take several seconds.
