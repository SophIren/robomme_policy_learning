"""Plan answer format for the full-plan VLM, shared by the dataset builder, preview and interactive scripts.

Pure python (json/re only) so it can be imported from any env (uv / robomme) by path.

Two formats:

native (default, close to Qwen3-VL grounding pretraining): the answer is a JSON list, one item per step, in plan
order; steps that act on an object carry its "point_2d" as [x, y] in 0..1000 relative coords (x first, normalized by
image width/height), the same coordinate convention as Qwen3-VL's own grounding output `[{"point_2d": [x, y], ...}]`.
Coordinates are plain numbers in the text, no `objects.bbox`. No "label" (object phrases repeat across steps, e.g.
"the right cube" twice) and no "step" index (the list order is the step order); both are still accepted when parsing.
    [
      {"subgoal": "pick up the red cube <point> for the first time", "point_2d": [469, 426]},
      {"subgoal": "drop the cube onto table"},
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
    "Output the full plan to complete the task as a JSON list of steps in order. Each step has a \"subgoal\"; "
    "if the step acts on an object, also give the object's \"point_2d\": [x, y]."
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
    for sg in grounded_plan:
        text, pts = split_grounded(sg)
        item = {"subgoal": text}
        if pts:
            rel = [px_to_rel(p, size) for p in pts]
            item["point_2d"] = rel[0] if len(rel) == 1 else rel
        items.append(item)
    return json.dumps(items, ensure_ascii=False)


# -----------------------------------------------------------------------------
# parsing an answer (GT row or model output), auto-detecting the format
# -----------------------------------------------------------------------------


def _strip_code_fence(text: str) -> str:
    m = re.search(r"```(?:json)?\s*(.*?)(?:```|$)", text, re.S)
    return m.group(1) if m else text


def _close_brackets(t: str) -> str:
    """Append the missing ']' / '}' of a truncated JSON (ignores brackets inside strings)."""
    stack, in_str, esc = [], False, False
    for ch in t:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch in "[{":
            stack.append("]" if ch == "[" else "}")
        elif ch in "]}" and stack:
            stack.pop()
    if in_str:
        t += '"'
    return t.rstrip().rstrip(",") + "".join(reversed(stack))


def _load_json(t: str):
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        pass
    # truncated output: cut back to the last complete "}" and close the open brackets,
    # so a half-written last step is dropped instead of being returned as garbage
    for end in range(len(t) - 1, 0, -1):
        if t[end] == "}":
            try:
                return json.loads(_close_brackets(t[: end + 1]))
            except json.JSONDecodeError:
                continue
    return None


def _is_pair(v) -> bool:
    return isinstance(v, list) and len(v) == 2 and all(isinstance(a, (int, float)) for a in v)


def _collect_pairs(v, out: list) -> None:
    """All [x, y] number pairs inside v (nested lists / dicts), in order."""
    if _is_pair(v):
        out.append(v)
    elif isinstance(v, list):
        for a in v:
            _collect_pairs(a, out)
    elif isinstance(v, dict):
        for a in v.values():
            _collect_pairs(a, out)


def _item_points(it: dict) -> list:
    """point_2d first; if the model put the coordinates under another key (e.g. "label": [x, y], or a nested
    list of {...: [x, y]}), take the number pairs from any field except "subgoal"."""
    pts: list = []
    if "point_2d" in it:
        _collect_pairs(it["point_2d"], pts)
    if not pts:
        for k, v in it.items():
            if k != "subgoal":
                _collect_pairs(v, pts)
    return pts


def _parse_native(text: str, size: int) -> list[dict] | None:
    data = _load_json(_strip_code_fence(text).strip())
    if data is None:
        return None
    if isinstance(data, dict):
        data = data.get("plan", data.get("steps", [data]))
    if not isinstance(data, list):
        return None
    steps = []
    for it in data:
        if not isinstance(it, dict) or "subgoal" not in it:
            continue  # only top-level items with a subgoal are steps
        tmpl = str(it["subgoal"]).strip()
        pts = [rel_to_px(q, size) for q in _item_points(it)]
        lab = it.get("label") if isinstance(it.get("label"), str) else None
        steps.append({"template": tmpl, "label": lab, "points": pts})
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
    """Rebuild the RoboMME grounded subgoal string the VLA was trained on: '... at <y, x> ...'.
    Placeholders without a predicted point are dropped (see missing_points)."""
    text = step["template"]
    for y, x in step["points"]:
        if POINT_TOKEN in text:
            text = text.replace(POINT_TOKEN, f"at <{y}, {x}>", 1)
        else:  # more points than placeholders: append, so the coordinate is not lost
            text = f"{text} at <{y}, {x}>"
    return re.sub(r"\s*" + re.escape(POINT_TOKEN), "", text).strip()


def missing_points(step: dict) -> int:
    """Number of <point> placeholders in the step that got no coordinate."""
    return max(0, step["template"].count(POINT_TOKEN) - len(step["points"]))
