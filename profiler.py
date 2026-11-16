import gc
import json
import logging
import os
import shutil
import time

import psutil
import torch
import yaml

import comfy_aimdo.control
import comfy.memory_management
import comfy.model_management
import comfy.nested_tensor
import comfy.system_memory
import execution
import folder_paths
import nodes
from comfy_api.latest import InputImpl

from . import render

NODE_ID = "VjumpkungMemoryProfiler"
SUBFOLDER = "vjumpkung_memory_profiler"
PAGE_SIZE = 32 << 20  # aimdo VBAR page
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")
DEFAULTS = {
    "name_prefix": "memory_profile",
    "require_node": True,
    "save_format": "both",
    "per_node": True,
    "block_residency": True,
    "cached_outputs": True,
    "tensor_scan": False,
}
PROCESS = psutil.Process()

_session = None
_hooks_installed = False


def load_config():
    if not os.path.exists(CONFIG_PATH):
        shutil.copyfile(CONFIG_PATH + ".example", CONFIG_PATH)
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return {**DEFAULTS, **(yaml.safe_load(f) or {})}


def memory_stats(torch_stats):
    device = comfy.model_management.get_torch_device()
    stats = dict.fromkeys(("vram_free", "vram_total", "aimdo_vram", "torch_allocated", "torch_reserved", "torch_peak"))
    stats["ram_total"] = comfy.system_memory.virtual_memory_total()
    stats["ram_available"] = comfy.system_memory.virtual_memory_available()
    stats["process_rss"] = PROCESS.memory_info().rss
    stats["pinned"] = comfy.model_management.TOTAL_PINNED_MEMORY
    if device.type == "cuda":
        stats["vram_free"], stats["vram_total"] = torch.cuda.mem_get_info(device)
        if comfy.memory_management.aimdo_enabled:
            stats["aimdo_vram"] = comfy_aimdo.control.get_total_vram_usage()
        if torch_stats:
            stats["torch_allocated"] = torch.cuda.memory_allocated(device)
            stats["torch_reserved"] = torch.cuda.memory_reserved(device)
            stats["torch_peak"] = torch.cuda.max_memory_allocated(device)
    return stats


def block_name(name):
    parts = name.split(".")
    for i, part in enumerate(parts):
        if part.isdigit():
            return ".".join(parts[:i + 1])
    return ".".join(parts[:2]) or "(root)"


def block_stats(patcher):
    model = patcher.model
    flags = {id(v): v.get_residency() for v in model.dynamic_vbars.values()} if patcher.is_dynamic() else {}
    blocks = {}
    for name, module in model.named_modules():
        tensors = [*module.parameters(recurse=False), *module.buffers(recurse=False)]
        size = sum(t.nbytes for t in tensors)
        if size == 0:
            continue
        v = module.__dict__.get("_v")
        if v is not None and v[2] and id(v[0]) in flags:
            vbar, address, length = v
            offset = address - vbar.base_addr
            first = offset // PAGE_SIZE
            count = (offset + length - 1) // PAGE_SIZE + 1 - first
            # pages past the end of the residency list are not resident
            resident = sum(1 for p in flags[id(vbar)][first:first + count] if p & 3)
            vram = size * resident // count
        else:
            vram = sum(t.nbytes for t in tensors if t.device.type not in ("cpu", "meta"))
        block = blocks.setdefault(block_name(name), {"name": block_name(name), "size": 0, "vram": 0})
        block["size"] += size
        block["vram"] += vram
    return list(blocks.values())


def model_stats(block_residency):
    models = []
    seen = set()
    for loaded in list(comfy.model_management.current_loaded_models):
        patcher = loaded.model
        if patcher is None or id(patcher.model) in seen or getattr(patcher.model, "comfy_writable", False):
            continue
        seen.add(id(patcher.model))
        entry = {
            "id": f"{id(patcher.model):x}",
            "name": type(patcher.model).__name__,
            "device": str(patcher.load_device),
            "dynamic": patcher.is_dynamic(),
            "size": patcher.model_size(),
            "loaded": patcher.loaded_size(),
            "pinned": patcher.pinned_memory_size(),
            "host_ram": patcher.loaded_ram_size(),
        }
        if block_residency:
            entry["blocks"] = block_stats(patcher)
        models.append(entry)
    return models


def iter_tensors(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for v in value.values():
            yield from iter_tensors(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            yield from iter_tensors(v)
    elif isinstance(value, comfy.nested_tensor.NestedTensor):
        yield from iter_tensors(value.tensors)
    elif isinstance(value, InputImpl.VideoFromComponents):  # VideoFromFile would decode on get_components
        c = value.get_components()
        yield from iter_tensors((c.images, c.alpha, c.audio))
    elif hasattr(value, "_comfy_cache_tensors"):
        yield from iter_tensors(value._comfy_cache_tensors())


def output_stats(executor, prompt):
    by_type = {}
    by_node = []
    seen = set()
    for node_id, node in prompt.items():
        entry = executor.caches.outputs.get_local(node_id)
        class_def = nodes.NODE_CLASS_MAPPINGS.get(node["class_type"])
        if entry is None or class_def is None:
            continue
        for io_type, slot in zip(class_def.RETURN_TYPES, entry.outputs):
            if isinstance(io_type, list):
                io_type = "COMBO"
            size = 0
            for t in iter_tensors(slot):
                storage = t.untyped_storage()
                key = (storage.data_ptr(), str(t.device))
                if key in seen or storage.nbytes() == 0:
                    continue
                seen.add(key)
                devices = by_type.setdefault(io_type, {})
                devices[str(t.device)] = devices.get(str(t.device), 0) + storage.nbytes()
                size += storage.nbytes()
            if size:
                by_node.append({"node_id": node_id, "class_type": node["class_type"], "type": io_type, "bytes": size})
    by_node.sort(key=lambda n: n["bytes"], reverse=True)
    by_type = dict(sorted(by_type.items(), key=lambda t: sum(t[1].values()), reverse=True))
    return {"by_type": by_type, "by_node": by_node}


def tensor_scan():
    seen = set()
    groups = {}
    for o in gc.get_objects():
        if type(o) not in (torch.Tensor, torch.nn.Parameter) or o.device.type == "meta" or o.layout != torch.strided:
            continue
        storage = o.untyped_storage()
        key = (storage.data_ptr(), str(o.device))
        if key in seen or storage.nbytes() == 0:
            continue
        seen.add(key)
        group = groups.setdefault((str(o.device), str(o.dtype)), {"device": str(o.device), "dtype": str(o.dtype), "count": 0, "bytes": 0})
        group["count"] += 1
        group["bytes"] += storage.nbytes()
    return sorted(groups.values(), key=lambda g: g["bytes"], reverse=True)


def snapshot(config, torch_stats):
    return {"memory": memory_stats(torch_stats), "models": model_stats(config["block_residency"])}


def make_report(kind, config, final, timeline):
    device = comfy.model_management.get_torch_device()
    return {
        "version": 1,
        "kind": kind,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": comfy.model_management.get_torch_device_name(device),
        "aimdo": comfy.memory_management.aimdo_enabled,
        "config": config,
        "timeline": timeline,
        "final": final,
    }


def profile_dir():
    return os.path.join(folder_paths.get_output_directory(), SUBFOLDER)


def list_profiles():
    folder = profile_dir()
    if not os.path.isdir(folder):
        return []
    files = [f for f in os.listdir(folder) if f.endswith(".json")]
    return sorted(files, key=lambda f: os.path.getmtime(os.path.join(folder, f)), reverse=True)


def save(report, save_format):
    folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(os.path.join(SUBFOLDER, report["config"]["name_prefix"]), folder_paths.get_output_directory())
    stem = f"{filename}_{counter:05}_"
    files = []
    if save_format != "png":
        with open(os.path.join(folder, stem + ".json"), "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        files.append(os.path.join(subfolder, stem + ".json"))
    if save_format != "json":
        render.render(report).save(os.path.join(folder, stem + ".png"))
        files.append(os.path.join(subfolder, stem + ".png"))
    logging.info("Memory Profiler: saved %s", ", ".join(files))
    return files


def manual_snapshot(save_format):
    config = load_config()
    final = snapshot(config, torch_stats=not comfy.memory_management.aimdo_enabled)
    if config["tensor_scan"]:
        final["tensors"] = tensor_scan()
    return save(make_report("manual", config, final, []), save_format)


def start_session(prompt):
    try:
        config = load_config()
        if config["require_node"] and not any(node.get("class_type") == NODE_ID for node in prompt.values()):
            return None
        device = comfy.model_management.get_torch_device()
        cuda = device.type == "cuda"
        if cuda:
            torch.cuda.reset_peak_memory_stats(device)
        return {"config": config, "device": device, "cuda": cuda, "timeline": [], "peak": 0, "start": time.perf_counter()}
    except Exception:
        logging.exception("Memory Profiler: failed to start profiling")
        return None


def record_node(session, node_id, class_type, status, seconds):
    try:
        entry = {"node_id": node_id, "class_type": class_type, "status": status, "seconds": seconds, **snapshot(session["config"], True)}
        session["timeline"].append(entry)
        if entry["memory"]["torch_peak"] is not None:
            session["peak"] = max(session["peak"], entry["memory"]["torch_peak"])
    except Exception:
        logging.exception("Memory Profiler: failed to record node %s", node_id)


def finish_session(session, executor, prompt, prompt_id):
    try:
        config = session["config"]
        final = snapshot(config, True)
        if config["cached_outputs"]:
            final["outputs"] = output_stats(executor, prompt)
        if config["tensor_scan"]:
            final["tensors"] = tensor_scan()
        if session["cuda"]:
            final["memory"]["torch_peak"] = max(session["peak"], final["memory"]["torch_peak"])
        report = make_report("run", config, final, session["timeline"])
        report.update(prompt_id=prompt_id, success=executor.success, seconds=time.perf_counter() - session["start"])
        save(report, config["save_format"])
    except Exception:
        logging.exception("Memory Profiler: failed to save profile")


def install_hooks():
    global _hooks_installed
    if _hooks_installed:
        return
    _hooks_installed = True
    run_prompt = execution.PromptExecutor.execute
    run_node = execution.execute

    def execute_prompt(executor, prompt, prompt_id, *args, **kwargs):
        global _session
        _session = start_session(prompt)
        try:
            return run_prompt(executor, prompt, prompt_id, *args, **kwargs)
        finally:
            session, _session = _session, None
            if session is not None:
                finish_session(session, executor, prompt, prompt_id)

    async def execute_node(server, dynprompt, caches, current_item, *args, **kwargs):
        session = _session
        if session is None or not session["config"]["per_node"]:
            return await run_node(server, dynprompt, caches, current_item, *args, **kwargs)
        if session["cuda"]:
            torch.cuda.reset_peak_memory_stats(session["device"])
        start = time.perf_counter()
        result = await run_node(server, dynprompt, caches, current_item, *args, **kwargs)
        if result[0] != execution.ExecutionResult.PENDING:
            record_node(session, current_item, dynprompt.get_node(current_item)["class_type"], result[0].name, time.perf_counter() - start)
        return result

    execution.PromptExecutor.execute = execute_prompt
    execution.execute = execute_node
