"""Interactive PoC: VLM predicts the full grounded plan once, you switch steps by hand while watching the sim.

Runs one or a few RoboMME episodes. A tiny web page shows the live front+wrist view, the plan with the current step
highlighted, the oracle subgoal as a hint, and buttons. Keys on the page: n/-> next, p/<- prev, space pause,
o single-step while paused, +/- speed, s stop episode.

terminal 0 (VLA server, symbolic grounded-subgoal policy):
    CUDA_VISIBLE_DEVICES=0 uv run scripts/serve_policy.py --seed=7 --port=8006 policy:checkpoint \
        --policy.dir=runs/ckpts/mme_vla_suite/symbolic-grounded-subgoal/79999 --policy.config=mme_vla_suite
terminal 1 (sim + VLM planner + web UI):
    micromamba activate robomme
    CUDA_VISIBLE_DEVICES=1 python examples/robomme/interactive_plan.py --port 8006 \
        --adapter runs/ckpts/vlm_plan_predictor/qwenvl/grounded_plan/<run>/checkpoint-XXX \
        --task MoveCube --episodes 0 --ui_port 8890
then open http://localhost:8890 (ssh -L 8890:localhost:8890 <cloud-host> if remote).
"""

import os

os.environ.setdefault("IMAGE_MAX_TOKEN_NUM", "256")
os.environ.setdefault("VIDEO_MAX_TOKEN_NUM", "64")
os.environ.setdefault("FPS_MAX_FRAMES", "10")

import argparse
import json
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2
import imageio
import numpy as np

from openpi_client import websocket_client_policy as _websocket_client_policy
from env_runner import EnvRunner
from utils import EpisodeState, RolloutRecorder, TASK_WITH_VIDEO_DEMO, pack_buffer

# Must match src/mme_vla_suite/dataset_builder/build_vlm_plan_dataset_qwenvl.py
PLAN_SYSTEM_PROMPT = (
    "You are a helpful assistant to help guide the robot to complete the task "
    "by predicting the full plan as a sequence of grounded language subgoals"
)
PLAN_USER_PROMPT = (
    "{video_prefix}The task goal is: {task_goal}\n"
    "<image>What's the full plan of grounded language subgoals "
    "to complete the task based on current observation?"
)
BOX_RE = re.compile(r"<\|box_start\|>\((\d+),(\d+)\)<\|box_end\|>")
POINT_RE = re.compile(r"<(\d+), (\d+)>")
IMAGE_SIZE = 256
BASE_MODEL = "Qwen/Qwen3-VL-4B-Instruct"


def load_engine(ckpt: str):
    """LoRA checkpoint (adapter_config.json) -> base + adapter; full fine-tune checkpoint -> load the dir itself."""
    from swift.llm import PtEngine

    if os.path.exists(os.path.join(ckpt, "adapter_config.json")):
        print(f"Loading {BASE_MODEL} + LoRA adapter {ckpt}")
        return PtEngine(model_id_or_path=BASE_MODEL, adapters=[ckpt], attn_impl="sdpa")
    print(f"Loading full fine-tuned model {ckpt}")
    return PtEngine(model_id_or_path=ckpt, attn_impl="sdpa")


# -----------------------------------------------------------------------------
# Planner
# -----------------------------------------------------------------------------


class PlanPredictor:
    def __init__(self, ckpt: str):
        self.engine = load_engine(ckpt)

    def predict(self, task_goal: str, image: np.ndarray, video: list | None, work_dir: Path) -> tuple[list[str], str]:
        from swift.llm import InferRequest, RequestConfig

        work_dir.mkdir(parents=True, exist_ok=True)
        image_path = str(work_dir / "plan_image.png")
        imageio.imwrite(image_path, image)
        req = {
            "messages": [
                {"role": "system", "content": PLAN_SYSTEM_PROMPT},
                {"role": "user", "content": PLAN_USER_PROMPT.format(
                    video_prefix="<video>" if video else "", task_goal=task_goal.lower())},
            ],
            "images": [image_path],
        }
        if video:
            video_path = str(work_dir / "plan_video.mp4")
            imageio.mimsave(video_path, video, fps=30)
            req["videos"] = [video_path]
        out = self.engine.infer([InferRequest(**req)], request_config=RequestConfig(max_tokens=512, temperature=0))
        raw = out[0].choices[0].message.content
        return parse_plan(raw), raw


def parse_plan(raw: str) -> list[str]:
    """'1. pick up the cube at <|box_start|>(y,x)<|box_end|>; 2. ...' -> ['pick up the cube at <Y, X>', ...] (pixels)."""
    text = BOX_RE.sub(
        lambda m: f"<{int(int(m.group(1)) * IMAGE_SIZE / 1000)}, {int(int(m.group(2)) * IMAGE_SIZE / 1000)}>", raw)
    return [re.sub(r"^\d+\.\s*", "", s.strip()) for s in text.split(";") if s.strip()]


# -----------------------------------------------------------------------------
# Shared state between sim loop and web UI
# -----------------------------------------------------------------------------


class UIState:
    def __init__(self):
        self.lock = threading.Lock()
        self.plan: list[str] = []
        self.idx = 0
        self.paused = True  # start paused so you can read the plan first
        self.single_step = False
        self.stop = False
        self.delay = 0.03  # seconds per sim step
        self.frame_jpg = b""
        self.info = {}
        self.switch_log: list[dict] = []
        self.flush = False  # drop queued actions so the new subgoal applies now

    def current(self) -> str | None:
        return self.plan[self.idx] if self.plan else None

    def command(self, c: str, q: dict, step: int) -> None:
        with self.lock:
            old = self.idx
            if c == "next" and self.idx < len(self.plan) - 1:
                self.idx += 1
            elif c == "prev" and self.idx > 0:
                self.idx -= 1
            elif c == "set" and "k" in q:
                self.idx = max(0, min(len(self.plan) - 1, int(q["k"][0])))
            elif c == "edit" and "text" in q:
                self.plan[self.idx] = q["text"][0]
                self.flush = True
                self.switch_log.append({"step": step, "edit": self.idx, "text": self.plan[self.idx]})
            elif c == "pause":
                self.paused = not self.paused
            elif c == "step":
                self.single_step = True
            elif c == "faster":
                self.delay = max(0.0, self.delay / 2 if self.delay > 0.004 else 0.0)
            elif c == "slower":
                self.delay = min(1.0, max(self.delay * 2, 0.005))
            elif c == "stop":
                self.stop = True
            if self.idx != old:
                self.flush = True
                self.switch_log.append({"step": step, "from": old, "to": self.idx, "subgoal": self.plan[self.idx]})


CSS = "\n".join([
    "body{font-family:sans-serif;margin:12px;background:#111;color:#ddd}",
    "img{image-rendering:pixelated;border:1px solid #444}",
    "li{padding:2px 4px;cursor:pointer}",
    "li.cur{background:#264;color:#fff;font-weight:bold}",
    "button{margin:2px;font-size:15px}",
    "#info{white-space:pre;font-family:monospace;font-size:13px}",
    ".row{display:flex;gap:16px}",
])

BUTTONS = "\n".join([
    "<button onclick=\"c('prev')\">&larr; prev (p)</button>",
    "<button onclick=\"c('next')\">next (n) &rarr;</button>",
    "<button onclick=\"c('pause')\">pause/run (space)</button>",
    "<button onclick=\"c('step')\">step (o)</button>",
    "<button onclick=\"c('slower')\">slower (-)</button>",
    "<button onclick=\"c('faster')\">faster (+)</button>",
    "<button onclick=\"c('stop')\">stop episode (s)</button>",
])

JS = """
function c(x,extra){fetch('/cmd?c='+x+(extra||''))}
document.addEventListener('keydown',e=>{if(e.target.tagName=='INPUT')return;
 const m={'n':'next','ArrowRight':'next','p':'prev','ArrowLeft':'prev',' ':'pause','o':'step','+':'faster','=':'faster','-':'slower','s':'stop'};
 if(m[e.key]){e.preventDefault();c(m[e.key])}});
setInterval(()=>{document.getElementById('f').src='/frame.jpg?'+Date.now()},100);
setInterval(async()=>{const s=await (await fetch('/state')).json();
 document.getElementById('plan').innerHTML=s.plan.map((p,i)=>`<li class="${i==s.idx?'cur':''}" onclick="c('set','&k=${i}')">${p.replace(/</g,'&lt;')}</li>`).join('');
 document.getElementById('info').textContent=JSON.stringify(s.info,null,1)},300);
"""

PAGE = (
    '<!doctype html><html><head><meta charset="utf-8"><title>plan PoC</title>'
    f"<style>{CSS}</style></head><body>"
    '<div class="row"><div><img id="f" width="1024"></div><div style="max-width:560px">'
    f"<div>{BUTTONS}</div>"
    '<div><input id="e" size="50" placeholder="override current step, e.g. pick up the cube at &lt;71, 94&gt;">'
    "<button onclick=\"c('edit','&text='+encodeURIComponent(document.getElementById('e').value))\">set</button></div>"
    '<ol id="plan" start="0"></ol><div id="info"></div></div></div>'
    f"<script>{JS}</script></body></html>"
)


def make_handler(ui: UIState, step_ref: dict):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, body: bytes, ctype: str):
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            u = urlparse(self.path)
            if u.path == "/":
                self._send(PAGE.encode(), "text/html")
            elif u.path == "/frame.jpg":
                with ui.lock:
                    jpg = ui.frame_jpg
                self._send(jpg, "image/jpeg")
            elif u.path == "/state":
                with ui.lock:
                    s = {"plan": ui.plan, "idx": ui.idx, "info": ui.info}
                self._send(json.dumps(s).encode(), "application/json")
            elif u.path == "/cmd":
                q = parse_qs(u.query)
                ui.command(q.get("c", [""])[0], q, step_ref["step"])
                self._send(b"ok", "text/plain")
            else:
                self.send_error(404)

    return H


def render(img: np.ndarray, wrist: np.ndarray, subgoal: str | None) -> bytes:
    front = img.copy()
    if subgoal:
        for y, x in POINT_RE.findall(subgoal):
            cv2.circle(front, (int(x), int(y)), 5, (255, 0, 255), -1)
    frame = np.concatenate([front, wrist], axis=1)
    frame = cv2.resize(frame, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
    ok, jpg = cv2.imencode(".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
    return jpg.tobytes()


# -----------------------------------------------------------------------------
# Episode loop (same control flow as examples/robomme/eval.py)
# -----------------------------------------------------------------------------


def get_action_chunk(client, epstate, img, wrist_img, robot_state, prompt, subgoal, exec_horizon, use_history):
    if use_history:
        resp = client.add_buffer(pack_buffer(epstate.image_buffer, epstate.state_buffer, epstate.exec_start_idx))
        while not resp.get("add_buffer_finished", False):
            time.sleep(0.1)
    element = {"observation/image": img, "observation/wrist_image": wrist_img,
               "observation/state": robot_state, "prompt": prompt}
    if subgoal is not None:
        element["simple_subgoal"] = subgoal
        element["grounded_subgoal"] = subgoal
    return client.infer(element)["actions"][:exec_horizon]


def run_episode(args, env_runner: EnvRunner, planner, ui: UIState, step_ref: dict, out_dir: Path) -> str:
    client = _websocket_client_policy.MMEVLAWebsocketClientPolicy(args.host, args.port)
    resp = client.reset()
    while not resp.get("reset_finished", False):
        time.sleep(0.1)

    pre = env_runner.get_init_obs()
    task_goal = pre["task_goal"]
    epstate = EpisodeState()
    epstate.image_buffer.extend(pre["images"])
    epstate.wrist_image_buffer.extend(pre["wrist_images"])
    epstate.state_buffer.extend(pre["states"])
    epstate.exec_start_idx = len(epstate.image_buffer) - 1

    ep_dir = out_dir / f"{env_runner.env_id}_ep{env_runner.episode_id}"
    recorder = RolloutRecorder(ep_dir, task_goal, fps=30)

    # ---- plan, once ----
    # video demo only for tasks that have one (training used it iff exec_start_idx > 0)
    has_demo = env_runner.env_id in TASK_WITH_VIDEO_DEMO and len(pre["images"]) > 1
    video = pre["images"][:-1] if has_demo else None
    if args.plan_file:
        plan, raw = json.load(open(args.plan_file)), "<from file>"
    else:
        print("predicting plan ...")
        plan, raw = planner.predict(task_goal, pre["images"][-1], video, ep_dir)
    print("RAW plan:", raw)
    for i, p in enumerate(plan):
        print(f"  {i}. {p}")

    with ui.lock:
        ui.plan, ui.idx, ui.paused, ui.stop, ui.switch_log = plan or ["(empty plan)"], 0, True, False, []
        ui.frame_jpg = render(pre["images"][-1], pre["wrist_images"][-1], ui.current())
        ui.info = {"task": env_runner.env_id, "episode": env_runner.episode_id, "goal": task_goal,
                   "has_video": video is not None, "status": "PAUSED - read the plan, press space to start"}

    img, wrist_img, robot_state = epstate.get_current_obs()
    success_flag = "unknown"
    while True:
        # wait while paused
        while True:
            with ui.lock:
                go = (not ui.paused) or ui.single_step or ui.stop
                ui.single_step = False
            if go:
                break
            time.sleep(0.05)
        with ui.lock:
            if ui.stop:
                success_flag = "stopped"
                break
            subgoal = ui.current()
            if ui.flush:
                epstate.action_plan.clear()  # re-query the VLA with the new subgoal right away
                ui.flush = False
            delay = ui.delay

        if not epstate.action_plan:
            chunk = get_action_chunk(client, epstate, img, wrist_img, robot_state, task_goal, subgoal,
                                     args.obs_horizon, args.use_history)
            epstate.action_plan.extend(chunk)
            epstate.clear_buffers()

        action = epstate.action_plan.popleft()
        obs, stop_flag, success_flag = env_runner.step(action)
        epstate.count += 1
        step_ref["step"] = epstate.count
        if obs[0] is None:
            break
        img, wrist_img, robot_state = obs
        epstate.add_observation(img, wrist_img, robot_state)
        recorder.record(image=img.copy(), wrist_image=wrist_img.copy(), state=robot_state.copy(),
                        action=action.copy(), subgoal=subgoal)

        with ui.lock:
            ui.frame_jpg = render(img, wrist_img, subgoal)
            ui.info.update({
                "step": epstate.count,
                "status": "paused" if ui.paused else f"running, {delay * 1000:.0f} ms/step",
                "current_step": f"{ui.idx}: {subgoal}",
                "oracle_hint": env_runner.info.get("grounded_subgoal_online"),
                "env_status": env_runner.info.get("status"),
            })

        if epstate.count > args.max_steps:
            success_flag = "timeout"
            break
        if stop_flag:
            break
        if delay:
            time.sleep(delay)

    with ui.lock:
        ui.info["status"] = f"FINISHED: {success_flag}"
        log = {"task": env_runner.env_id, "episode": env_runner.episode_id, "task_goal": task_goal,
               "result": success_flag, "steps": epstate.count, "raw_plan": raw, "plan": ui.plan,
               "switches": ui.switch_log}
    ep_dir.mkdir(parents=True, exist_ok=True)
    json.dump(log, open(ep_dir / "log.json", "w"), indent=1)
    recorder.save_video(f"{success_flag}.mp4")
    print(f"[{env_runner.env_id} ep{env_runner.episode_id}] {success_flag} after {epstate.count} steps -> {ep_dir}")
    return success_flag


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8006, help="VLA policy server port")
    p.add_argument("--adapter", default=None, help="plan VLM checkpoint dir (LoRA adapter or full fine-tune)")
    p.add_argument("--plan_file", default=None, help="json list of subgoals instead of the VLM (debug switching)")
    p.add_argument("--task", default="MoveCube")
    p.add_argument("--exclude_tasks", default="ButtonUnmaskSwap,PickHighlight",
                   help="tasks the plan model does not support yet; refused unless --force")
    p.add_argument("--force", action="store_true", help="run an excluded task anyway")
    p.add_argument("--episodes", default="0", help="comma-separated episode ids")
    p.add_argument("--dataset", default="test", choices=["train", "val", "test"])
    p.add_argument("--max_steps", type=int, default=1300)
    p.add_argument("--obs_horizon", type=int, default=16)
    p.add_argument("--no_history", action="store_true")
    p.add_argument("--ui_port", type=int, default=8890)
    p.add_argument("--ui_host", default="127.0.0.1")
    p.add_argument("--out_dir", default="runs/evaluation/plan_poc")
    args = p.parse_args()
    args.use_history = not args.no_history
    assert args.adapter or args.plan_file, "pass --adapter or --plan_file"
    if args.task in {t for t in args.exclude_tasks.split(",") if t} and not args.force:
        raise SystemExit(f"{args.task} is excluded for the plan model (--exclude_tasks); pass --force to run it anyway")

    planner = PlanPredictor(args.adapter) if args.adapter else None
    ui, step_ref = UIState(), {"step": 0}
    server = ThreadingHTTPServer((args.ui_host, args.ui_port), make_handler(ui, step_ref))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"UI: http://{args.ui_host}:{args.ui_port}")

    out_dir = Path(args.out_dir)
    results = {}
    env_runner = EnvRunner(args.task, out_dir, max_steps=args.max_steps, dataset=args.dataset)
    for ep in [int(e) for e in args.episodes.split(",")]:
        env_runner.make_env(ep)
        try:
            results[ep] = run_episode(args, env_runner, planner, ui, step_ref, out_dir)
        finally:
            env_runner.close_env()
    print("results:", results)
    input("done; press Enter to close the UI ")
    server.shutdown()


if __name__ == "__main__":
    main()
