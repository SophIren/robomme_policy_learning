"""Plan answer format for the full-plan VLM, shared by the dataset builder, preview and interactive scripts.

Pure python (json/re only) so it can be imported from any env (uv / robomme) by path.

Two formats:

native (default, close to Qwen3-VL grounding pretraining): the answer is a JSON list, one item per step;
steps that act on an object carry the object's "label" and its "point_2d" as [x, y] in 0..1000 relative coords
(x first, normalized by image width/height), exactly like Qwen3-VL's own grounding output
`[{"point_2d": [x, y], "label": "..."}]`. Coordinates are plain numbers in the text, no `objects.bbox`.
    [
      {"step": 1, "subgoal": "pick up the red cube <point> for the first time", "label": "the red cube", "point_2d": [367, 426]},
      {"step": 2, "subgoal": "drop the cube onto table"},
      ...
    ]
  "<point>" marks where "at <y, x>" stood in the original grounded subgoal, so the VLA string can be rebuilt exactly.

legacy (first version): "1. pick up the red cube at <bbox> ...; 2. ..." + `objects.bbox` [[y, x]] in 0..255,
  rendered by ms-swift as <|box_start|>(y,x)<|box_end|>.

Pixel convention everywhere outside the model: [y, x] in 0..IMAGE_SIZE-1 (same as RoboMME grounded subgoals).
"""

from __future__ import annotations

import json
import re

IMAGE_SIZE = 256
POINT_TOKEN = "<point>"
GROUNDED_RE = re.compile(r"\s*at <(\d+), (\d+)>")
LEGACY_BOX_RE = re.compile(r"<\|box_start\|>\((\d+),(\d+)\)<\|box_end\|>")
# last "the ..." phrase before the point: "place the red cube onto the target at <..>" -> "the target"
LABEL_RE = re.compile(r"(the (?:(?!\bthe\b)[a-z\- ])+?)\s*" + re.escape(POINT_TOKEN))

NATIVE_INSTRUCTION = (
    "Output the full plan to complete the task as a JSON list of steps. Each step has a \"subgoal\"; "
    "if the step acts on an object, also give the object's \"label\" and \"point_2d\": [x, y]."
)
LEGACY_SYSTEM_PROMPT = (
    "You are a helpful assistant to help guide the robot to complete the task "
    "by predicting the full plan as a sequence of grounded language subgoals"
)
LEGACY_QUESTION = "What's the full plan of grounded language subgoals to complete the task based on current observation?"


# -----------------------------------------------------------------------------
# coordinates
# -----------------------------------------------------------------------------


def px_to_rel(yx: list[int], size: int = IMAGE_SIZE) -> list[int]:
    """[y, x] pixels -> [x, y] in 0..1000 (Qwen3-VL relative)."""
    y, x = yx
    return [int(round(x * 1000 / size)), int(round(y * 1000 / size))]


def rel_to_px(xy: list[float], size: int = IMAGE_SIZE) -> list[int]:
    """[x, y] in 0..1000 -> [y, x] pixels, clipped to the image."""
    x, y = xy
    clip = lambda v: max(0, min(size - 1, int(round(v * size / 1000))))
    return [clip(y), clip(x)]


# -----------------------------------------------------------------------------
# prompts
# -----------------------------------------------------------------------------


def build_messages(task_goal: str, has_video: bool, fmt: str = "native") -> list[dict]:
    """Messages without the assistant answer (system message only for the legacy format)."""
    video = "<video>" if has_video else ""
    if fmt == "native":
        # no custom system prompt: keep the default chat template, as in grounding pretraining
        return [{"role": "user", "content": f"{video}<image>The task goal is: {task_goal}\n{NATIVE_INSTRUCTION}"}]
    return [
        {"role": "system", "content": LEGACY_SYSTEM_PROMPT},
        {"role": "user", "content": f"{video}The task goal is: {task_goal}\n<image>{LEGACY_QUESTION}"},
    ]


def prompt_messages(row: dict) -> list[dict]:
    """Messages of a dataset row without the final assistant answer."""
    return [m for m in row["messages"] if m["role"] != "assistant"]


def user_text(row: dict) -> str:
    return next(m["content"] for m in row["messages"] if m["role"] == "user")


# -----------------------------------------------------------------------------
# building the answer (dataset builder)
# -----------------------------------------------------------------------------


def split_grounded(subgoal: str) -> tuple[str, list[list[int]]]:
    """'pick up the cube at <71, 94> now' -> ('pick up the cube <point> now', [[71, 94]])."""
    pts = [[int(a), int(b)] for a, b in GROUNDED_RE.findall(subgoal)]
    text = GROUNDED_RE.sub(" " + POINT_TOKEN, subgoal).strip()
    return text, pts


def label_of(template: str) -> str | None:
    found = LABEL_RE.findall(template)
    return found[0].strip() if found else None


def native_answer(grounded_plan: list[str], size: int = IMAGE_SIZE) -> str:
    items = []
    for i, sg in enumerate(grounded_plan):
        text, pts = split_grounded(sg)
        item = {"step": i + 1, "subgoal": text}
        if pts:
            lab = label_of(text)
            if lab:
                item["label"] = lab
            rel = [px_to_rel(p, size) for p in pts]
            item["point_2d"] = rel[0] if len(rel) == 1 else rel
        items.append(item)
    return json.dumps(items, ensure_ascii=False)


# -----------------------------------------------------------------------------
# parsing an answer (GT row or model output), auto-detecting the format
# -----------------------------------------------------------------------------


def _strip_code_fence(text: str) -> str:
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    return m.group(1) if m else text


def _parse_native(text: str, size: int) -> list[dict] | None:
    t = _strip_code_fence(text).strip()
    try:
        data = json.loads(t)
    except json.JSONDecodeError:
        # truncated / slightly broken output: keep the complete {...} items
        data = []
        for m in re.finditer(r"\{[^{}]*\}", t):
            try:
                data.append(json.loads(m.group(0)))
            except json.JSONDecodeError:
                pass
        if not data:
            return None
    if isinstance(data, dict):
        data = [data]
    steps = []
    for it in data:
        if not isinstance(it, dict):
            continue
        tmpl = str(it.get("subgoal", "")).strip()
        p = it.get("point_2d")
        pts = []
        if isinstance(p, list) and p:
            pts = [p] if not isinstance(p[0], list) else p
            pts = [rel_to_px(q, size) for q in pts if len(q) == 2]
        steps.append({"template": tmpl, "label": it.get("label"), "points": pts})
    return steps


def _parse_legacy(text: str, pts_px: list[list[int]] | None, size: int) -> list[dict]:
    """'1. a at <bbox>; 2. b' (+ GT pts) or model output with <|box_start|>(y,x)<|box_end|> (0..1000)."""
    if pts_px is None:
        pts_px = [[int(int(a) * size / 1000), int(int(b) * size / 1000)] for a, b in LEGACY_BOX_RE.findall(text)]
        text = LEGACY_BOX_RE.sub("<bbox>", text)
    it = iter(pts_px)
    steps = []
    for s in text.split(";"):
        s = re.sub(r"^\d+\.\s*", "", s.strip())
        if not s:
            continue
        n = s.count("<bbox>")
        tmpl = re.sub(r"\s*at <bbox>", " " + POINT_TOKEN, s).replace("<bbox>", POINT_TOKEN)
        steps.append({"template": tmpl, "label": label_of(tmpl),
                      "points": [p for p in (next(it, None) for _ in range(n)) if p is not None]})
    return steps


def parse_answer(text: str, gt_points: list[list[int]] | None = None, size: int = IMAGE_SIZE) -> list[dict]:
    """-> [{"template": str with <point>, "label": str|None, "points": [[y, x] px, ...]}, ...]."""
    t = text.strip()
    if t.startswith("[") or t.startswith("{") or t.startswith("```") or "point_2d" in t:
        steps = _parse_native(t, size)
        if steps is not None:
            return steps
    return _parse_legacy(t, gt_points, size)


def parse_row(row: dict, size: int = IMAGE_SIZE) -> list[dict]:
    answer = next(m["content"] for m in reversed(row["messages"]) if m["role"] == "assistant")
    gt_pts = row.get("objects", {}).get("bbox") if "objects" in row else None
    return parse_answer(answer, gt_pts, size)


def to_vla_subgoal(step: dict) -> str:
    """Rebuild the RoboMME grounded subgoal string the VLA was trained on: '... at <y, x> ...'."""
    text = step["template"]
    for y, x in step["points"]:
        text = text.replace(POINT_TOKEN, f"at <{y}, {x}>", 1)
    return re.sub(r"\s*" + re.escape(POINT_TOKEN), "", text).strip()
