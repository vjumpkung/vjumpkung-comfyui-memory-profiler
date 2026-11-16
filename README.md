# vjumpkung ComfyUI Memory Profiler

Records VRAM, RAM and per-block model residency for every ComfyUI run and saves the result as JSON and a PNG report.

## What is recorded

- After every executed node: VRAM used/total, aimdo VRAM, torch allocated/reserved/peak, RAM used/total, process RSS and pinned memory, plus every loaded model with its size, loaded bytes, pinned bytes, host RAM and per-block VRAM.
- At the end of the run: the same snapshot, the memory held by cached node outputs grouped by type (`IMAGE`, `LATENT`, `CONDITIONING`, ...), and optionally every live torch tensor grouped by device and dtype.

The PNG has a header, a timeline chart, a residency heatmap per model (columns are nodes in execution order, blue is RAM, green is VRAM, grey is not loaded) and the output and tensor tables.

## Usage

Add a **Memory Profiler** node (`utils/memory_profiler`) to the workflow. It has no inputs or outputs; its presence enables profiling for that run. Files are written to `output/vjumpkung_memory_profiler/<name_prefix>_00001_.json` and `.png`.

Right-click the node for an instant snapshot (JSON, PNG or both) of the current memory state. Snapshots are refused while a prompt is running.

**Memory Profile to Image** renders a saved JSON profile to an image and writes the PNG next to it. Press `R` to refresh the profile list after new files appear.

## Config

`config.yaml` is created from `config.yaml.example` on first run and re-read at the start of every prompt.

| key | default | meaning |
| --- | --- | --- |
| `name_prefix` | `memory_profile` | file name prefix |
| `require_node` | `true` | only profile prompts that contain a Memory Profiler node |
| `save_format` | `both` | `json`, `png` or `both` |
| `per_node` | `true` | snapshot after every executed node |
| `block_residency` | `true` | per-block VRAM of loaded models in each snapshot |
| `cached_outputs` | `true` | memory held by node outputs at the end of the run |
| `tensor_scan` | `false` | scan all live torch tensors at the end of the run and in manual snapshots (slow) |

## JSON layout

```
version, kind ("run" | "manual"), created, device, aimdo, config
prompt_id, success, seconds                  (runs only)
timeline: [{node_id, class_type, status, seconds, memory, models}]
final: {memory, models, outputs?, tensors?}
memory: {vram_free, vram_total, aimdo_vram, torch_allocated, torch_reserved, torch_peak,
         ram_total, ram_available, process_rss, pinned}
models: [{id, name, device, dynamic, size, loaded, pinned, host_ram, blocks?: [{name, size, vram}]}]
outputs: {by_type: {type: {device: bytes}}, by_node: [{node_id, class_type, type, bytes}]}
tensors: [{device, dtype, count, bytes}]
```

Values that are unavailable on the current device are `null`.

## Caveats

- Hooks wrap `execution.PromptExecutor.execute` and `execution.execute`.
- With `per_node` the torch peak counter is reset before every node, so `torch_peak` in the timeline is the peak inside that node. Torch counters are not available in manual snapshots when aimdo is enabled.
- Dynamic VRAM block residency is read from aimdo VBAR pages (32 MiB granularity). Block `vram` is the stored weight size scaled by the resident fraction, so weights that are cast to a smaller dtype on load (for example fp32 weights run as fp16) report more than the real VRAM use. Compare with the model's `loaded` value.
- On Windows (WDDM) `mem_get_info` reports usage for the whole GPU, not just this process.
- Cached output sizes cover top-level node outputs only. Tensors shared between outputs are counted once.
- `tensor_scan` walks every Python object and can take several seconds.
