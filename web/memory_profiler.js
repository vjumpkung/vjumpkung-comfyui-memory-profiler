import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

async function snapshot(format) {
    const response = await api.fetchApi("/vjumpkung_memory_profiler/snapshot", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ format }),
    });
    const data = await response.json().catch(() => ({}));
    if (response.ok) {
        app.extensionManager.toast.add({
            severity: "success",
            summary: "Memory snapshot saved",
            detail: data.files.join("\n"),
            life: 5000,
        });
    } else {
        app.extensionManager.toast.add({
            severity: "warn",
            summary: "Memory snapshot failed",
            detail: data.error ?? response.statusText,
            life: 5000,
        });
    }
}

app.registerExtension({
    name: "vjumpkung.MemoryProfiler",
    getNodeMenuItems(node) {
        if (node.constructor.comfyClass !== "VjumpkungMemoryProfiler") return [];
        return [
            { content: "Memory snapshot (JSON)", callback: () => snapshot("json") },
            { content: "Memory snapshot (PNG)", callback: () => snapshot("png") },
            { content: "Memory snapshot (JSON + PNG)", callback: () => snapshot("both") },
        ];
    },
});
