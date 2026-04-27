#!/usr/bin/env python3
"""
Convert G1 motion pkl files to LeRobot v2 format dataset.

Uses the official LeRobot API (LeRobotDataset.create / add_frame / save_episode).

Usage:
    python convert_to_lerobot_v2.py --input_dir G1_motion_data --output_dir G1_lerobot_v2

Dependencies:
    pip install lerobot imageio imageio-ffmpeg
"""

import os
import sys
import glob
import shutil
import argparse
import joblib
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from visualize_stickman import StickmanVisualizer, KEYPOINT_INDICES

from lerobot.datasets.lerobot_dataset import LeRobotDataset


FPS = 50
IMAGE_SHAPE = (224, 224, 3)
CAMERAS = ["head_img", "left_wrist_img", "right_wrist_img"]
REPO_ID = "lerobot_datasets/g1_motion"


FEATURES = {
    "observation.state": {
        "dtype": "float32",
        "shape": (36,),
        "names": (
            ["root_trans_x", "root_trans_y", "root_trans_z"]
            + ["root_rot_w", "root_rot_x", "root_rot_y", "root_rot_z"]
            + [f"joint_{i}" for i in range(29)]
        ),
    },
    "action": {
        "dtype": "float32",
        "shape": (36,),
        "names": (
            ["root_trans_x", "root_trans_y", "root_trans_z"]
            + ["root_rot_w", "root_rot_x", "root_rot_y", "root_rot_z"]
            + [f"joint_{i}" for i in range(29)]
        ),
    },
    "task_vis_stickman": {
        "dtype": "float32",
        "shape": (900,),
        "names": None,
    },
    **{
        f"observation.images.{cam}": {
            "dtype": "video",
            "shape": IMAGE_SHAPE,
            "names": ["height", "width", "channel"],
        }
        for cam in CAMERAS
    },
}

PLACEHOLDER_IMAGE = np.zeros(IMAGE_SHAPE, dtype=np.uint8)


def _build_state(data: dict) -> np.ndarray:
    root_trans = np.asarray(data["reset_root_trans"], dtype=np.float32)  # (T, 3)
    root_rot = np.asarray(data["reset_root_rot"], dtype=np.float32)  # (T, 4)
    joint_pos = np.asarray(data["reset_joint_pos"], dtype=np.float32)  # (T, 29)
    return np.concatenate([root_trans, root_rot, joint_pos], axis=1)  # (T, 36)


def convert(input_dir: str, output_dir: str, fps: int = FPS, visualize: bool = False) -> None:
    pkl_files = sorted(glob.glob(os.path.join(input_dir, "*.pkl")))
    output_path = Path(output_dir)
    if output_path.exists():
        shutil.rmtree(output_path)

    dataset = LeRobotDataset.create(
        repo_id=REPO_ID,
        robot_type="unitree_g1_29dof",
        fps=fps,
        features=FEATURES,
        root=output_path,
        image_writer_threads=4,
    )

    for ep_idx, pkl_path in enumerate(pkl_files):
        print(f"[{ep_idx + 1}/{len(pkl_files)}] {os.path.basename(pkl_path)}")

        data = joblib.load(pkl_path)
        state = _build_state(data)                                                       # (T, 36)
        if state.shape[0] < 100:
            print(f"  → skipped (only {state.shape[0]} frames)")
            continue

        root_origin = state[0, :3].copy()                                                # 固定首帧 root 坐标，避免别名问题
        state[:, :3] -= root_origin                                                      # 平移归零：所有帧减去第一帧的 x,y,z
        action = np.concatenate([state[1:], state[-1:]], axis=0)                         # (T, 36)，t+1时刻state作为action，末帧复用

        body_pos = np.asarray(data["body_pos"], dtype=np.float32)                        # (T, 23, 3)，复用已加载的数据
        stickman = body_pos[:, KEYPOINT_INDICES, :]                                      # (T, 6, 3)
        stick_origin = stickman[0, 0, :].copy()                                          # 固定首帧骨盆坐标，避免别名问题
        stickman -= stick_origin                                                         # 平移归零：所有节点减去首帧骨盆(root)坐标
        stickman = stickman.reshape(len(stickman), 18)                                   # (T, 18)
        sample_idx = np.linspace(0, len(stickman) - 1, 50, dtype=int)
        stickman = stickman[sample_idx]                                                  # (50, 18)

        if visualize:
            kp_xyz = stickman.reshape(len(stickman), 6, 3)
            viz = StickmanVisualizer(kp_xyz, title=os.path.basename(pkl_path), fps=fps)
            viz.run()

        task_name = os.path.basename(pkl_path)
        T = state.shape[0]
        for t in range(T):
            frame = {
                "observation.state": state[t],
                "action": action[t],
                "task_vis_stickman": stickman.reshape(-1),
            }
            for cam in CAMERAS:
                frame[f"observation.images.{cam}"] = PLACEHOLDER_IMAGE

            dataset.add_frame(frame, task="")

        dataset.save_episode()
        print(f"  → episode {ep_idx:06d}  |  {T} frames")

    print(f"\nDone! {len(pkl_files)} episodes saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="G1_1000")
    parser.add_argument("--output_dir", default="G1_1000_lerobot")
    parser.add_argument("--fps", type=int, default=FPS)
    args = parser.parse_args()

    convert(args.input_dir, args.output_dir, args.fps)
