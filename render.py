import math

import numpy as np
from matplotlib import colormaps
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
from PIL import Image

GB = 1 << 30
WIDTH = 13  # inches, 1300 px at DPI
DPI = 100
LINE = 0.16  # inches per text line or heatmap row
RESIDENCY = colormaps["viridis"].with_extremes(bad="lightgrey")

VRAM_SERIES = (("process VRAM", "process_vram"), ("torch allocated", "torch_allocated"), ("torch peak (in node)", "torch_peak"))
RAM_SERIES = (("process RAM (RSS)", "process_rss"), ("pinned RAM", "pinned"))


def _fmt(n):
    if n is None:
        return "-"
    if n >= GB:
        return f"{n / GB:.2f} GB"
    return f"{n / (1 << 20):.2f} MB"


def _clip(text, limit=60):
    return text if len(text) <= limit else text[:limit - 3] + "..."


def _header_lines(report):
    memory = report["final"]["memory"]
    lines = [f"{report['created']}   device: {report['device']}   aimdo: {'on' if report['aimdo'] else 'off'}"]
    if report["kind"] == "run":
        lines.append(f"prompt {report['prompt_id']}   {'success' if report['success'] else 'failed'}   {report['seconds']:.2f}s   {len(report['timeline'])} nodes")
    if memory["vram_total"] is not None:
        lines.append(f"VRAM  process {_fmt(memory['process_vram'])} of {_fmt(memory['vram_total'])}   aimdo {_fmt(memory['aimdo_vram'])}   torch allocated {_fmt(memory['torch_allocated'])}   reserved {_fmt(memory['torch_reserved'])}   peak {_fmt(memory['torch_peak'])}")
    lines.append(f"RAM   process {_fmt(memory['process_rss'])} of {_fmt(memory['ram_total'])}   pinned {_fmt(memory['pinned'])}")
    return lines


def _header(subfig, report, lines):
    ax = subfig.subplots()
    ax.axis("off")
    ax.set_title(f"Memory profile ({report['kind']}), ComfyUI process only", loc="left", fontsize=14)
    ax.text(0, 1, "\n".join(lines), transform=ax.transAxes, va="top", family="monospace", fontsize=9)


def _timeline(subfig, timeline, cuda):
    charts = [("VRAM", VRAM_SERIES)] if cuda else []
    charts.append(("RAM", RAM_SERIES))
    steps = range(1, len(timeline) + 1)
    for ax, (title, series) in zip(subfig.subplots(len(charts), 1, squeeze=False)[:, 0], charts):
        for label, key in series:
            ax.plot(steps, [entry["memory"][key] / GB for entry in timeline], marker="o" if len(timeline) <= 100 else None, markersize=3, label=label)
        ax.set_title(f"{title} used by the ComfyUI process", loc="left", fontsize=10)
        ax.set_xlabel("node (execution order)")
        ax.set_ylabel("GB")
        ax.set_xlim(0.5, len(timeline) + 0.5)
        ax.set_ylim(bottom=0)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(alpha=0.3)
        ax.legend(loc="upper left", fontsize=8)


def _nodes(subfig, timeline):
    ax = subfig.subplots()
    ax.axis("off")
    lines = [_clip(f"{i + 1}. #{entry['node_id']} {entry['class_type']} {entry['status']} {entry['seconds']:.2f}s") for i, entry in enumerate(timeline)]
    rows = -(-len(lines) // 3)
    for column in range(3):
        ax.text(column / 3, 1, "\n".join(lines[column * rows:(column + 1) * rows]), transform=ax.transAxes, va="top", family="monospace", fontsize=8)


def _residency(model_id, present):
    entries = [models.get(model_id) for models in present]
    first = next(e for e in entries if e)
    blocks = [{b["name"]: b for b in e.get("blocks", [])} if e else {} for e in entries]
    names = list(dict.fromkeys(name for column in blocks for name in column))
    if not names:
        return first, entries[-1], [f"(whole model)  {_fmt(first['size'])}"], [[e["loaded"] / e["size"] if e and e["size"] else np.nan for e in entries]]
    labels = [_clip(f"{name}  {_fmt(next(column[name]['size'] for column in blocks if name in column))}") for name in names]
    ratios = [[column[name]["vram"] / column[name]["size"] if name in column else np.nan for column in blocks] for name in names]
    return first, entries[-1], labels, ratios


def _heatmaps(subfig, heatmaps):
    axes = subfig.subplots(len(heatmaps), 1, squeeze=False, height_ratios=[len(labels) for _, _, labels, _ in heatmaps])[:, 0]
    for ax, (first, last, labels, ratios) in zip(axes, heatmaps):
        image = ax.imshow(np.array(ratios), aspect="auto", interpolation="nearest", cmap=RESIDENCY, vmin=0, vmax=1)
        columns = len(ratios[0])
        ticks = range(columns - 1, -1, -math.ceil(columns / 40))
        ax.set_xticks(ticks, ["end" if i == columns - 1 else str(i + 1) for i in ticks], fontsize=7)
        ax.set_yticks(range(len(labels)), labels, fontsize=7)
        title = f"{first['name']}  {'dynamic' if first['dynamic'] else 'static'}  {first['device']}  size {_fmt(first['size'])}"
        title += f"   end: loaded {_fmt(last['loaded'])}  pinned {_fmt(last['pinned'])}  host RAM {_fmt(last['host_ram'])}" if last else "   end: not loaded"
        ax.set_title(title, loc="left", fontsize=9)
    subfig.colorbar(image, ax=axes, fraction=0.015, pad=0.01, label="fraction in VRAM (0 = RAM)")
    subfig.suptitle("Model residency   columns: node execution order, end = final   grey: not loaded", x=0.01, ha="left", fontsize=11)


def _table(ax, title, columns, rows, widths):
    ax.axis("off")
    ax.set_title(title, loc="left", fontsize=10)
    table = ax.table(cellText=rows, colLabels=columns, colWidths=widths, cellLoc="left", bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(8)


def _outputs(subfig, outputs):
    by_type = outputs["by_type"]
    types = list(by_type)
    bar_ax, table_ax = subfig.subplots(1, 2, width_ratios=[3, 2])
    left = np.zeros(len(types))
    for device in dict.fromkeys(d for devices in by_type.values() for d in devices):
        sizes = np.array([by_type[t].get(device, 0) / GB for t in types])
        bars = bar_ax.barh(types, sizes, left=left, label=device)
        left += sizes
    bar_ax.bar_label(bars, [_fmt(sum(by_type[t].values())) for t in types], padding=3, fontsize=8)
    bar_ax.invert_yaxis()
    bar_ax.margins(x=0.2)
    bar_ax.set_xlabel("GB")
    bar_ax.set_title("Cached node outputs", loc="left", fontsize=10)
    bar_ax.legend(fontsize=8)
    rows = [(f"#{n['node_id']}", n["class_type"], n["type"], _fmt(n["bytes"])) for n in outputs["by_node"][:10]]
    _table(table_ax, "Largest nodes", ("node", "class", "type", "size"), rows, (0.1, 0.42, 0.28, 0.2))


def _tensors(subfig, groups):
    rows = [(g["device"], g["dtype"], g["count"], _fmt(g["bytes"])) for g in groups[:20]]
    _table(subfig.subplots(), "Live tensors", ("device", "dtype", "count", "size"), rows, (0.3, 0.3, 0.2, 0.2))


def render(report):
    final = report["final"]
    timeline = report["timeline"]
    cuda = final["memory"]["vram_total"] is not None
    lines = _header_lines(report)
    panels = [(0.5 + LINE * len(lines), _header, report, lines)]
    if timeline:
        panels.append((0.4 + 2.6 * (2 if cuda else 1), _timeline, timeline, cuda))
        panels.append((0.2 + LINE * -(-len(timeline) // 3), _nodes, timeline))
    present = [{m["id"]: m for m in snap["models"]} for snap in [*timeline, final]]
    heatmaps = [_residency(i, present) for i in dict.fromkeys(i for models in present for i in models)]
    if heatmaps:
        panels.append((0.5 + sum(0.75 + LINE * len(labels) for _, _, labels, _ in heatmaps), _heatmaps, heatmaps))
    if final.get("outputs", {}).get("by_node"):
        outputs = final["outputs"]
        panels.append((0.6 + 0.22 * max(len(outputs["by_type"]), min(10, len(outputs["by_node"])) + 1), _outputs, outputs))
    if final.get("tensors"):
        panels.append((0.5 + 0.22 * (min(20, len(final["tensors"])) + 1), _tensors, final["tensors"]))
    figure = Figure(figsize=(WIDTH, sum(p[0] for p in panels)), dpi=DPI, layout="constrained")
    subfigs = figure.subfigures(len(panels), 1, squeeze=False, height_ratios=[p[0] for p in panels])[:, 0]
    for subfig, (_, draw, *args) in zip(subfigs, panels):
        draw(subfig, *args)
    canvas = FigureCanvasAgg(figure)
    canvas.draw()
    return Image.fromarray(np.asarray(canvas.buffer_rgba())).convert("RGB")
