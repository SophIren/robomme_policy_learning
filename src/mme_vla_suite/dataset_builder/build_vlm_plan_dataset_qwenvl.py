"""
Build VLM *full plan* prediction dataset for QwenVL (PoC).

Unlike build_vlm_subgoal_dataset_qwenvl.py (one next subgoal per frame, with
history), here one sample = one episode:
    input : [video demo] + task goal + first execution frame
    output: numbered plan of grounded subgoals, e.g.
            "1. pick up the cube at <bbox>; 2. drop the cube onto table; ..."
Pixel coordinates are passed through ms-swift `objects.bbox` (same convention as
the existing grounded-subgoal dataset), in order of appearance in the answer.
"""

import json
import os

import h5py
import imageio

from mme_vla_suite.dataset_builder.robomme_h5_utils import (
    get_task_goal,
    get_timestep_indices,
    preprocess_grounded_subgoal,
    wrap_history_subgoals,
    add_noise_to_bbox,
)
from mme_vla_suite.dataset_builder.vlm_subgoal_dataset_base import BaseVLMSubgoalDatasetBuilder

PLAN_SYSTEM_PROMPT = (
    "You are a helpful assistant to help guide the robot to complete the task "
    "by predicting the full plan as a sequence of grounded language subgoals"
)


class DatasetBuilder(BaseVLMSubgoalDatasetBuilder):
    def __init__(
        self,
        *args,
        frame_offsets: tuple[int, ...] = (0,),
        bbox_noise: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        # augment: several early frames (robot has barely moved) -> same plan
        self.frame_offsets = frame_offsets
        self.bbox_noise = bbox_noise
        self.plan_train_data_path = os.path.join(self.data_dir, "plan_train.jsonl")
        if os.path.exists(self.plan_train_data_path):
            os.remove(self.plan_train_data_path)

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
        video_prefix = "<video>" if video_path else ""
        texts, bboxes = [], []
        for sg in plan:
            text, bbox = preprocess_grounded_subgoal(sg)
            texts.append(text)
            bboxes.extend(bbox)

        user_prompt = (
            f"{video_prefix}The task goal is: {task_goal}\n"
            "<image>What's the full plan of grounded language subgoals "
            "to complete the task based on current observation?"
        )
        result = {
            "messages": [
                {"role": "system", "content": PLAN_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": wrap_history_subgoals(texts)},
            ],
            "objects": {
                "ref": [],
                "bbox": add_noise_to_bbox(bboxes) if self.bbox_noise else bboxes,
            },
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
