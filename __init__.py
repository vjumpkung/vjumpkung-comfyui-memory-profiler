import json
import os

import numpy as np
import torch
from aiohttp import web
from typing_extensions import override

import server
from comfy_api.latest import ComfyExtension, io, ui

from . import profiler, render

WEB_DIRECTORY = "web"


class MemoryProfiler(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=profiler.NODE_ID,
            display_name="Memory Profiler",
            category="utils/memory_profiler",
            description="Records a memory profile of every run into output/vjumpkung_memory_profiler. Right-click for an instant snapshot.",
            inputs=[],
            outputs=[],
        )

    @classmethod
    def execute(cls) -> io.NodeOutput:
        return io.NodeOutput()


class MemoryProfileToImage(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="VjumpkungMemoryProfileToImage",
            display_name="Memory Profile to Image",
            category="utils/memory_profiler",
            description="Renders a saved memory profile JSON as an image.",
            is_output_node=True,
            inputs=[io.Combo.Input("profile", options=profiler.list_profiles())],
            outputs=[io.Image.Output()],
        )

    @classmethod
    def execute(cls, profile) -> io.NodeOutput:
        if profile not in profiler.list_profiles():
            raise ValueError(f"Unknown memory profile: {profile}")
        with open(os.path.join(profiler.profile_dir(), profile), encoding="utf-8") as f:
            image = render.render(json.load(f))
        png = os.path.splitext(profile)[0] + ".png"
        image.save(os.path.join(profiler.profile_dir(), png))
        tensor = torch.from_numpy(np.array(image.convert("RGB")).astype(np.float32) / 255.0)[None]
        return io.NodeOutput(tensor, ui=ui.SavedImages([ui.SavedResult(png, profiler.SUBFOLDER, io.FolderType.output)]))


@server.PromptServer.instance.routes.post("/vjumpkung_memory_profiler/snapshot")
async def snapshot(request):
    save_format = (await request.json()).get("format")
    if save_format not in ("json", "png", "both"):
        return web.json_response({"error": "format must be json, png or both"}, status=400)
    if server.PromptServer.instance.prompt_queue.currently_running:
        return web.json_response({"error": "Snapshots are unavailable while a prompt is running"}, status=409)
    return web.json_response({"files": profiler.manual_snapshot(save_format)})


class MemoryProfilerExtension(ComfyExtension):
    @override
    async def on_load(self) -> None:
        profiler.install_hooks()

    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [MemoryProfiler, MemoryProfileToImage]


async def comfy_entrypoint() -> MemoryProfilerExtension:
    return MemoryProfilerExtension()
