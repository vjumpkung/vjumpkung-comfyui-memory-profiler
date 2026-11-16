import math

from PIL import Image, ImageDraw, ImageFont

FONT = ImageFont.load_default(size=14)
TITLE = ImageFont.load_default(size=20)

WIDTH = 1300
GAP = 12
PAD = 12
ROW = 16

BG = (24, 26, 31)
FG = (225, 228, 235)
DIM = (140, 146, 158)
GRID = (52, 56, 66)
ABSENT = (58, 62, 72)
RAM = (66, 133, 244)
VRAM = (76, 175, 80)
OTHER = (255, 193, 7)


def _vram_used(memory):
    if memory["vram_total"] is None:
        return None
    return memory["vram_total"] - memory["vram_free"]


SERIES = (
    ("VRAM used", VRAM, _vram_used),
    ("aimdo VRAM", OTHER, lambda m: m["aimdo_vram"]),
    ("torch peak (in node)", (239, 83, 80), lambda m: m["torch_peak"]),
    ("torch allocated", (171, 71, 188), lambda m: m["torch_allocated"]),
    ("process RSS", RAM, lambda m: m["process_rss"]),
)


def _fmt(n):
    if n is None:
        return "-"
    if n >= 1 << 30:
        return f"{n / (1 << 30):.2f} GB"
    return f"{n / (1 << 20):.2f} MB"


def _canvas(width, height):
    image = Image.new("RGB", (width, height), BG)
    return image, ImageDraw.Draw(image)


def _lerp(a, b, t):
    return tuple(round(x + (y - x) * t) for x, y in zip(a, b))


def _clip(text, limit):
    while len(text) > 4 and FONT.getlength(text) > limit:
        text = text[:-4] + "..."
    return text


def _label_step(slot):
    return max(1, math.ceil(28 / slot))


def _header(report):
    memory = report["final"]["memory"]
    ram_used = memory["ram_total"] - memory["ram_available"]
    lines = [f"{report['created']}   device: {report['device']}   aimdo: {'on' if report['aimdo'] else 'off'}"]
    if report["kind"] == "run":
        lines.append(f"prompt {report['prompt_id']}   {'success' if report['success'] else 'failed'}   {report['seconds']:.2f}s   {len(report['timeline'])} nodes")
    lines.append(f"VRAM   used {_fmt(_vram_used(memory))} / {_fmt(memory['vram_total'])}   aimdo {_fmt(memory['aimdo_vram'])}   torch allocated {_fmt(memory['torch_allocated'])}   reserved {_fmt(memory['torch_reserved'])}   peak {_fmt(memory['torch_peak'])}")
    lines.append(f"RAM    used {_fmt(ram_used)} / {_fmt(memory['ram_total'])}   process {_fmt(memory['process_rss'])}   pinned {_fmt(memory['pinned'])}")
    image, draw = _canvas(WIDTH, 2 * PAD + 30 + ROW * len(lines))
    draw.text((PAD, PAD), f"Memory profile ({report['kind']})", font=TITLE, fill=FG)
    for i, line in enumerate(lines):
        draw.text((PAD, PAD + 30 + ROW * i), line, font=FONT, fill=FG)
    return image


def _timeline(timeline):
    count = len(timeline)
    series = []
    for label, color, get in SERIES:
        values = [get(entry["memory"]) for entry in timeline]
        if any(v is not None for v in values):
            series.append((label, color, values))
    chart = 300 if series else 0
    rows = -(-count // 3)
    image, draw = _canvas(WIDTH, chart + PAD + ROW * rows + PAD)
    if series:
        x0, x1, y0, y1 = 90, WIDTH - 20, 36, chart - 28
        peak = max(v for _, _, values in series for v in values if v is not None) or 1
        for i in range(5):
            y = y1 - (y1 - y0) * i // 4
            draw.line([(x0, y), (x1, y)], fill=GRID)
            draw.text((x0 - 8, y), _fmt(peak * i / 4), font=FONT, fill=DIM, anchor="rm")
        slot = (x1 - x0) / count
        step = _label_step(slot)
        for i in range(count - 1, -1, -step):
            draw.text((x0 + slot * (i + 0.5), y1 + 6), str(i + 1), font=FONT, fill=DIM, anchor="ma")
        legend_x = x0
        for label, color, values in series:
            points = [(x0 + slot * (i + 0.5), y1 - (y1 - y0) * v / peak) for i, v in enumerate(values) if v is not None]
            if len(points) > 1:
                draw.line(points, fill=color, width=2)
            if count <= 100:
                for x, y in points:
                    draw.ellipse([x - 3, y - 3, x + 3, y + 3], fill=color)
            draw.rectangle([legend_x, 12, legend_x + 12, 24], fill=color)
            draw.text((legend_x + 18, 10), label, font=FONT, fill=FG)
            legend_x += 18 + FONT.getlength(label) + 24
    column = (WIDTH - 2 * PAD) // 3
    for i, entry in enumerate(timeline):
        text = f"{i + 1}. #{entry['node_id']} {entry['class_type']} {entry['status']} {entry['seconds']:.2f}s"
        draw.text((PAD + column * (i % 3), chart + PAD + ROW * (i // 3)), _clip(text, column - 8), font=FONT, fill=FG)
    return image


def _legend():
    image, draw = _canvas(WIDTH, 2 * PAD + 30 + ROW + 4)
    draw.text((PAD, PAD), "Model residency", font=TITLE, fill=FG)
    y = PAD + 30
    draw.text((PAD, y), "columns: node execution order, end = final", font=FONT, fill=DIM)
    x = PAD + 340
    draw.text((x, y), "RAM", font=FONT, fill=FG)
    x += 40
    for i in range(160):
        draw.line([(x + i, y + 1), (x + i, y + ROW - 2)], fill=_lerp(RAM, VRAM, i / 159))
    draw.text((x + 170, y), "VRAM", font=FONT, fill=FG)
    draw.rectangle([x + 230, y + 1, x + 244, y + ROW - 2], fill=ABSENT)
    draw.text((x + 252, y), "not loaded", font=FONT, fill=FG)
    return image


def _heatmap(model_id, present):
    entries = [models.get(model_id) for models in present]
    first = next(e for e in entries if e)
    last = entries[-1]
    blocks = [{b["name"]: b for b in e.get("blocks", [])} if e else {} for e in entries]
    names = list(dict.fromkeys(name for column in blocks for name in column))
    rows = []
    if names:
        for name in names:
            size = next(column[name]["size"] for column in blocks if name in column)
            ratios = [column[name]["vram"] / column[name]["size"] if name in column else None for column in blocks]
            rows.append((f"{name}  {_fmt(size)}", ratios))
    else:
        ratios = [e["loaded"] / e["size"] if e and e["size"] else None for e in entries]
        rows.append((f"(whole model)  {_fmt(first['size'])}", ratios))

    label_width = min(420, int(max(FONT.getlength(label) for label, _ in rows)) + 12)
    cell = max(4, min(200, (WIDTH - label_width - 2 * PAD) // len(entries)))
    image, draw = _canvas(max(WIDTH, label_width + 2 * PAD + cell * len(entries)), PAD + ROW + ROW + ROW * len(rows) + PAD)
    title = f"{first['name']}  {'dynamic' if first['dynamic'] else 'static'}  {first['device']}  size {_fmt(first['size'])}"
    if last:
        title += f"   end: loaded {_fmt(last['loaded'])}  pinned {_fmt(last['pinned'])}  host RAM {_fmt(last['host_ram'])}"
    else:
        title += "   end: not loaded"
    draw.text((PAD, PAD // 2), title, font=FONT, fill=FG)
    x0 = PAD + label_width
    step = _label_step(cell)
    for i in range(len(entries) - 1, -1, -step):
        draw.text((x0 + cell * i, PAD + ROW - 2), "end" if i == len(entries) - 1 else str(i + 1), font=FONT, fill=DIM)
    for r, (label, ratios) in enumerate(rows):
        y = PAD + 2 * ROW + ROW * r
        draw.text((PAD, y), _clip(label, label_width - 12), font=FONT, fill=FG)
        for c, ratio in enumerate(ratios):
            color = ABSENT if ratio is None else _lerp(RAM, VRAM, min(1, ratio))
            draw.rectangle([x0 + cell * c, y, x0 + cell * (c + 1) - (2 if cell >= 6 else 1), y + ROW - 2], fill=color)
    return image


def _heatmaps(report):
    present = [{m["id"]: m for m in snap["models"]} for snap in [*report["timeline"], report["final"]]]
    model_ids = list(dict.fromkeys(i for models in present for i in models))
    if not model_ids:
        return []
    return [_legend(), *(_heatmap(i, present) for i in model_ids)]


def _outputs(outputs):
    by_type = outputs["by_type"]
    top = outputs["by_node"][:10]
    largest = max((sum(d.values()) for d in by_type.values()), default=1)
    image, draw = _canvas(WIDTH, 2 * PAD + 30 + ROW * len(by_type) + 2 * ROW + ROW * len(top))
    draw.text((PAD, PAD), "Cached node outputs", font=TITLE, fill=FG)
    y = PAD + 30
    bar_x, bar_width = PAD + 130, 500
    for io_type, devices in by_type.items():
        draw.text((PAD, y), io_type, font=FONT, fill=FG)
        x = bar_x
        for device, size in devices.items():
            width = max(1, round(bar_width * size / largest))
            draw.rectangle([x, y + 1, x + width - 1, y + ROW - 2], fill=RAM if device == "cpu" else VRAM)
            x += width
        breakdown = "  ".join(f"{device} {_fmt(size)}" for device, size in devices.items())
        draw.text((bar_x + bar_width + 12, y), f"{_fmt(sum(devices.values()))}   ({breakdown})", font=FONT, fill=FG)
        y += ROW
    draw.text((PAD, y + ROW // 2), "largest nodes", font=FONT, fill=DIM)
    y += ROW + ROW // 2
    for node in top:
        draw.text((PAD, y), f"#{node['node_id']} {node['class_type']}  {node['type']}  {_fmt(node['bytes'])}", font=FONT, fill=FG)
        y += ROW
    return image


def _tensors(groups):
    groups = groups[:20]
    image, draw = _canvas(WIDTH, 2 * PAD + 30 + ROW * (len(groups) + 1))
    draw.text((PAD, PAD), "Live tensors", font=TITLE, fill=FG)
    columns = (PAD, PAD + 140, PAD + 340, PAD + 460)
    y = PAD + 30
    for x, text in zip(columns, ("device", "dtype", "count", "size")):
        draw.text((x, y), text, font=FONT, fill=DIM)
    for group in groups:
        y += ROW
        for x, text in zip(columns, (group["device"], group["dtype"], str(group["count"]), _fmt(group["bytes"]))):
            draw.text((x, y), text, font=FONT, fill=FG)
    return image


def render(report):
    final = report["final"]
    sections = [_header(report)]
    if report["timeline"]:
        sections.append(_timeline(report["timeline"]))
    sections += _heatmaps(report)
    if "outputs" in final:
        sections.append(_outputs(final["outputs"]))
    if "tensors" in final:
        sections.append(_tensors(final["tensors"]))
    image = Image.new("RGB", (max(s.width for s in sections), sum(s.height for s in sections) + GAP * (len(sections) - 1)), BG)
    y = 0
    for section in sections:
        image.paste(section, (0, y))
        y += section.height + GAP
    return image
