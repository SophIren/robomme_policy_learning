"""
Build VLM *full plan* prediction dataset for QwenVL (PoC).

Unlike build_vlm_subgoal_dataset_qwenvl.py (one next subgoal per frame, with
history), here one sample = one episode:
    input : [video demo] + task goal + first execution frame
    output: the whole plan of grounded subgoals.

Answer formats (see plan_format.py):
- native (default): JSON list of steps with "label" and "point_2d": [x, y] in 0..1000,
  i.e. Qwen3-VL's own grounding output format; coordinates are plain text, no `objects`.
- legacy: "1. pick up the cube at <bbox>; 2. ..." + ms-swift `objects.bbox` [[y, x]] in 0..255
  (rendered as <|box_start|>(y,x)<|box_end|>), the first version of this dataset.
"""

import json
import os

import h5py
import imageio

from mme_vla_suite.dataset_builder import plan_format as pf
from mme_vla_suite.dataset_builder.robomme_h5_utils import (
    add_noise_to_bbox,
    get_task_goal,
    get_timestep_indices,
    preprocess_grounded_subgoal,
    wrap_history_subgoals,
)
from mme_vla_suite.dataset_builder.vlm_subgoal_dataset_base import BaseVLMSubgoalDatasetBuilder


class DatasetBuilder(BaseVLMSubgoalDatasetBuilder):
    def __init__(
        self,
        *args,
        answer_format: str = "native",
        frame_offsets: tuple[int, ...] = (0,),
        bbox_noise: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        assert answer_format in ("native", "legacy"), answer_format
        self.answer_format = answer_format
        # augment: several early frames (robot has barely moved) -> same plan
        self.frame_offsets = frame_offsets
        self.bbox_noise = bbox_noise
        self.plan_train_data_path = os.path.join(self.data_dir, "plan_train.jsonl")
        if os.path.exists(self.plan_train_data_path):
            os.remove(self.plan_train_data_path)
        print(f"plan answer format: {answer_format} -> {self.plan_train_data_path}")

    def _extract_plan(
        self, episode_data: h5py.Group, exec_start_idx: int, num_timesteps: int
    ) -> list[str]:
        """Grounded subgoal at the start of every simple-subgoal segment."""
        plan: list[str] = []
        last_simple = None
        for idx in range(exec_start_idx, num_timesteps):
            info = episode_data[f"timestep_{idx}"]["info"]
            simple = info["simple_subgoal"][()].decode().lower()
            if "complete" in simple:
                continue
            if simple != last_simple:
                grounded = info["grounded_subgoal"][()].decode().lower()
                if "complete" not in grounded:
                    plan.append(grounded)
                last_simple = simple
        return plan

    def make_plan_data(
        self,
        task_goal: str,
        plan: list[str],
        image_path: str,
        video_path: str | None,
    ) -> dict:
        if self.answer_format == "native":
            if self.bbox_noise:
                plan = [_jitter_grounded(sg) for sg in plan]
            messages = pf.build_messages(task_goal, video_path is not None, "native")
            messages.append({"role": "assistant", "content": pf.native_answer(plan)})
            result = {"messages": messages, "images": [image_path]}
        else:
            texts, bboxes = [], []
            for sg in plan:
                text, bbox = preprocess_grounded_subgoal(sg)
                texts.append(text)
                bboxes.extend(bbox)
            messages = pf.build_messages(task_goal, video_path is not None, "legacy")
            messages.append({"role": "assistant", "content": wrap_history_subgoals(texts)})
            result = {
                "messages": messages,
                "objects": {"ref": [], "bbox": add_noise_to_bbox(bboxes) if self.bbox_noise else bboxes},
                "images": [image_path],
            }
        if video_path:
            result["videos"] = [video_path]
        return result

    def process_per_episode(
        self,
        env_dataset: h5py.File,
        env_id: str,
        episode_idx: int,
    ) -> None:
        print(f"processing episode {episode_idx} of {env_id}...")
        episode_data = env_dataset[f"episode_{episode_idx}"]
        task_goal = get_task_goal(episode_data, lower=True)
        num_timesteps = len(get_timestep_indices(episode_data))
        exec_start_idx = self._first_execution_step(episode_data)

        plan = self._extract_plan(episode_data, exec_start_idx, num_timesteps)
        if not plan:
            print(f"  empty plan, skip {env_id} ep{episode_idx}")
            return

        video_path = None
        if exec_start_idx > 0:
            video_frames = [
                episode_data[f"timestep_{i}"]["obs"]["front_rgb"][()]
                for i in range(exec_start_idx)
            ]
            video_path = os.path.join(
                self.images_dir, f"{env_id}_ep{episode_idx}_video.mp4"
            )
            imageio.mimsave(video_path, video_frames, fps=30)

        for off in self.frame_offsets:
            idx = min(exec_start_idx + off, num_timesteps - 1)
            image = episode_data[f"timestep_{idx}"]["obs"]["front_rgb"][()]
            image_path = os.path.join(
                self.images_dir, f"{env_id}_ep{episode_idx}_step{idx}.png"
            )
            imageio.imwrite(image_path, image)
            row = self.make_plan_data(task_goal, plan, image_path, video_path)
            with open(self.plan_train_data_path, "a") as f:
                f.write(json.dumps(row) + "\n")


def _jitter_grounded(subgoal: str) -> str:
    """Same +-2 px noise as the legacy bbox noise, applied to 'at <y, x>' in the text."""
    text, pts = pf.split_grounded(subgoal)
    for y, x in add_noise_to_bbox(pts):
        text = text.replace(pf.POINT_TOKEN, f"at <{y}, {x}>", 1)
    return text
