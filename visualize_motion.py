
import os
import time
import json
import joblib
import mujoco
import glob
import argparse
import mujoco.viewer
import numpy as np
from typing import Dict, Any, List, Optional
from scipy.spatial.transform import Rotation as R

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colormaps
import scipy.ndimage.filters as scipy_filters
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import imageio

HUMAN_BODY_LINKS = [
    'pelvis', # 0
                   
    'left_hip_link', 'left_knee_link', 'left_foot_link', 'left_toe_link', # 1, 2, 3, 4

    'right_hip_link', 'right_knee_link', 'right_foot_link', 'right_toe_link', # 5, 6, 7, 8

    'spine0_link', 'spine1_link', 'spine2_link', # 9, 10, 11

    'left_thorax_link', 'left_shoulder_link', 'left_elbow_link', 'left_wrist_link', # 12, 13, 14, 15

    'right_thorax_link', 'right_shoulder_link', 'right_elbow_link', 'right_wrist_link', # 16, 17, 18, 19

    'neck0_link', 'neck1_link', 'head_link' # 20, 21, 22
]

HUMAN_BODY_LINKS_PARENT_MAP = {
    "pelvis": "world",

    'left_hip_link': "pelvis",
    'left_knee_link': "left_hip_link",
    'left_foot_link': "left_knee_link",
    'left_toe_link': "left_foot_link",

    'right_hip_link': "pelvis",
    'right_knee_link': "right_hip_link",
    'right_foot_link': "right_knee_link",
    'right_toe_link': "right_foot_link",

    'spine0_link': "pelvis",
    'spine1_link': "spine0_link",
    'spine2_link': "spine1_link",

    'left_thorax_link': "spine2_link",
    'left_shoulder_link': "left_thorax_link",
    'left_elbow_link': "left_shoulder_link",
    'left_wrist_link': "left_elbow_link",

    'right_thorax_link': "spine2_link",
    'right_shoulder_link': "right_thorax_link",
    'right_elbow_link': "right_shoulder_link",
    'right_wrist_link': "right_elbow_link",

    'neck0_link': "spine2_link",
    'neck1_link': "neck0_link",
    'head_link': "neck1_link",
}

HUMAN_KEYPOINTS_LINKS = [
    'pelvis', 
                         
    'left_hip_link', 'left_knee_link', 'left_foot_link',

    'right_hip_link', 'right_knee_link', 'right_foot_link',

    'left_shoulder_link', 'left_elbow_link', 'left_wrist_link', 

    'right_shoulder_link', 'right_elbow_link', 'right_wrist_link', 
]

HUMAN_BODY_LINKS_COLOR = [
    "1 0 0 0.30",        # pelvis - 红色

    "0.68 0.85 0.9 1", # left_hip_link - 浅天蓝色
    "0.47 0.70 0.80 1", # left_knee_link - 中天蓝色
    "0.30 0.50 0.65 1", # left_foot_link - 深天蓝色
    "0.20 0.40 0.50 1", # left_toe_link - 更深的天蓝色

    "0.74 1 0.74 1",   # right_hip_link - 浅薄荷绿
    "0.55 0.89 0.55 1", # right_knee_link - 中薄荷绿
    "0.40 0.70 0.40 1", # right_foot_link - 深薄荷绿
    "0.30 0.60 0.30 1", # right_toe_link - 更深的薄荷绿

    "1 1 0.6 1",       # spine0_link - 浅黄色
    "1 1 0.4 1",       # spine1_link - 中黄色
    "1 1 0.2 1",       # spine2_link - 深黄色

    "0.82 0.68 1 1",   # left_thorax_link - 浅紫罗兰色
    "0.75 0.5 1 1",    # left_shoulder_link - 中紫罗兰色
    "0.65 0.4 0.85 1", # left_elbow_link - 深紫罗兰色
    "0.55 0.3 0.75 1", # left_wrist_link - 更深的紫罗兰色

    "1 0.8 0.6 1",     # right_thorax_link - 浅橘色
    "1 0.71 0.59 1",   # right_shoulder_link - 中橘色
    "1 0.6 0.4 1",     # right_elbow_link - 深橘色
    "1 0.5 0.3 1",     # right_wrist_link - 更深的橘色

    "1 0.75 0.85 1",   # neck0_link - 浅粉红色
    "1 0.6 0.75 1",    # neck1_link - 中粉红色
    "1 0.45 0.65 1"    # head_link - 深粉红色
]

HUMAN_KEYPOINTS_PARENT_MAP = {
    "pelvis": "world",

    'left_hip_link': "pelvis",
    'left_knee_link': "left_hip_link",
    'left_foot_link': "left_knee_link",

    'right_hip_link': "pelvis",
    'right_knee_link': "right_hip_link",
    'right_foot_link': "right_knee_link",

    'left_shoulder_link': "pelvis",
    'left_elbow_link': "left_shoulder_link",
    'left_wrist_link': "left_elbow_link",

    'right_shoulder_link': "pelvis",
    'right_elbow_link': "right_shoulder_link",
    'right_wrist_link': "right_elbow_link",
}

def add_bounded_noise(kp_xyz, min_mag=0.020, max_mag=0.045, seed=None):

    if seed is not None:
        np.random.seed(seed)
    
    J = kp_xyz.shape[0]
    
    noise_dir = np.random.randn(J, 3)
    norms = np.linalg.norm(noise_dir, axis=1, keepdims=True)
    noise_dir = noise_dir / norms
    
    magnitudes = np.random.uniform(min_mag, max_mag, size=(J, 1))
    
    noise = noise_dir * magnitudes  # (J, 3)
    
    noisy_kp_xyz = kp_xyz + noise
    
    return noisy_kp_xyz

class Data_Loader:

    @staticmethod
    def load_motion_data(motion_path, data_type="pkl"):
        
        data = {}

        if data_type == "pkl":

            if motion_path.endswith('.pkl'):
                key = os.path.splitext(os.path.basename(motion_path))[0]
                motion_data = joblib.load(motion_path)
                data[key] = motion_data
                return data
            
            files = [f for f in glob.glob(os.path.join(motion_path, "**", "*.pkl"), recursive=True) if f.endswith('.pkl')]
            
            for fname in files:
                key = os.path.splitext(os.path.relpath(fname, motion_path))[0]
                try:
                    motion_data = joblib.load(fname)
                    data[key] = motion_data
                except Exception as e:
                    print(f"Error loading {fname}: {e}")

        elif data_type == "json":

            if motion_path.endswith('.json'):
                key = os.path.splitext(os.path.basename(motion_path))[0]
                with open(motion_path, 'r') as f:
                    motion_data = json.load(f)
                poses = motion_data.pop("poses")
                motion_len = len(poses)
                poses = np.asarray(poses).reshape(motion_len, -1, 7)
                motion_data["body_pos"] = poses[:, :, :3]
                motion_data["body_rot"] = poses[:, :, 3:]
                data[key] = motion_data
                return data
            
            files = [f for f in glob.glob(os.path.join(motion_path, "**", "*.json"), recursive=True) if f.endswith('.json')]
            
            for file in files:
                key = os.path.splitext(os.path.relpath(file, motion_path))[0]
                try:
                    with open(file, 'r') as f:
                        motion_data = json.load(f)
                    poses = motion_data.pop("poses")
                    motion_len = len(poses)
                    poses = np.asarray(poses).reshape(motion_len, -1, 7)
                    motion_data["body_pos"] = poses[:, :, :3]
                    motion_data["body_rot"] = poses[:, :, 3:]
                    data[key] = motion_data
                except Exception as e:
                    print(f"Error loading {file}: {e}")

        return data
    
class MotionVisualizer:

    def __init__(self, data: Dict[str, Dict[str, Any]], bad_dir: str, fps: float = 50.0):
        self.data = data
        self.bad_dir = bad_dir

        self.play_fps = float(fps)
        self.is_paused = False
        self.is_closed = False
        self._switch_request = 0

        self.show_rotation = True
        self.show_grig = True

        self.keys: List[str] = list(self.data.keys())
        self.data_len = len(self.keys)
        self.curr_data_id = 0

        self.curr_key: Optional[str] = None
        self.xyz: Optional[np.ndarray] = None   # (T,J,3)
        self.wxyz: Optional[np.ndarray] = None  # (T,J,4)
        self.num_frames = 0
        self.current_frame = 0

        self.keypoints_indices = [HUMAN_BODY_LINKS.index(l) for l in HUMAN_KEYPOINTS_LINKS]

        self.body_links_parent_indices, self.keypoints_parent_indices = self._compute_parent_indices()
        self.body_bone_pairs = [(i, p) for i, p in enumerate(self.body_links_parent_indices) if p != -1]
        self.kp_bone_pairs = [(i, p) for i, p in enumerate(self.keypoints_parent_indices) if p != -1]

        self.fig = plt.figure(figsize=(24, 16))
        self.ax_full = self.fig.add_subplot(121, projection="3d")
        self.ax_kp = self.fig.add_subplot(122, projection="3d")

        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self.fig.canvas.mpl_connect("close_event", self._on_close)

        self._setup_axis(self.ax_full)
        self._setup_axis(self.ax_kp)

        self._init_plot()

        if self.data_len > 0:
            self._load_current_data()
            self._render(frame_idx=0, force_draw=True)

    def _on_close(self, event):
        self.is_closed = True

    def _on_key(self, event):
        k = (event.key or "").lower()

        if k == " ":
            self.is_paused = not self.is_paused

        elif k == "r":
            self.current_frame = 0

        elif k in ["n", "right"]:
            self._switch_request = +1

        elif k in ["u", "left"]:
            self._switch_request = -1

        elif k == "b":
            if self.curr_key is not None:
                target = os.path.join(self.bad_dir, f"{self.curr_key}.pkl")
                os.makedirs(os.path.dirname(target), exist_ok=True)
                joblib.dump(self.data[self.curr_key], target)

        elif k in ["+", "="]:
            self._set_play_fps(self.play_fps * 1.25)
        elif k in ["-", "_"]:
            self._set_play_fps(self.play_fps / 1.25)
        elif k == "0":
            self._set_play_fps(50.0)

        elif k in [".", ">"]:
            if self.num_frames > 0:
                self.is_paused = True
                self.current_frame = (self.current_frame + 1) % self.num_frames
                self._render(self.current_frame, force_draw=True)
        elif k in [",", "<"]:
            if self.num_frames > 0:
                self.is_paused = True
                self.current_frame = (self.current_frame - 1) % self.num_frames
                self._render(self.current_frame, force_draw=True)

        elif k == "t":
            self.show_rotation = not self.show_rotation

            if self.num_frames > 0:
                self._render(self.current_frame, force_draw=True)

        elif k == "g":
            self.show_grig = not self.show_grig

        elif k in ["q", "escape"]:
            self.is_closed = True
            plt.close(self.fig)

    def _set_play_fps(self, new_fps: float):
        new_fps = float(new_fps)
        new_fps = max(1.0, min(new_fps, 120.0))
        self.play_fps = new_fps

    def _load_current_data(self):
        if self.data_len == 0:
            self.curr_key = None
            self.xyz = None
            self.wxyz = None
            self.num_frames = 0
            self.current_frame = 0
            return

        self.curr_data_id %= self.data_len
        self.curr_key = self.keys[self.curr_data_id]
        curr = self.data[self.curr_key]

        self.xyz = np.asarray(curr["body_pos"])   # (T,J,3)
        self.wxyz = np.asarray(curr["body_rot"]) #+ 1e-12  # (T,J,4) wxyz

        self.num_frames = int(self.xyz.shape[0])
        self.current_frame = 0

    def _compute_parent_indices(self):
        body_parent = []
        for link in HUMAN_BODY_LINKS:
            parent = HUMAN_BODY_LINKS_PARENT_MAP.get(link, "world")
            while parent not in HUMAN_BODY_LINKS and parent != "world":
                parent = HUMAN_BODY_LINKS_PARENT_MAP.get(parent, "world")
            body_parent.append(-1 if parent == "world" else HUMAN_BODY_LINKS.index(parent))

        kp_parent = []
        for link in HUMAN_KEYPOINTS_LINKS:
            parent = HUMAN_KEYPOINTS_PARENT_MAP.get(link, "world")
            while parent not in HUMAN_KEYPOINTS_LINKS and parent != "world":
                parent = HUMAN_KEYPOINTS_PARENT_MAP.get(parent, "world")
            kp_parent.append(-1 if parent == "world" else HUMAN_KEYPOINTS_LINKS.index(parent))

        return body_parent, kp_parent

    def _setup_axis(self, ax):
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.grid(self.show_grig)

    def _init_plot(self):
        # full body
        self.full_bones = Line3DCollection([], colors="black", linewidths=1.5, alpha=0.6)
        self.full_x = Line3DCollection([], colors="red", linewidths=1.2)
        self.full_y = Line3DCollection([], colors="green", linewidths=1.2)
        self.full_z = Line3DCollection([], colors="blue", linewidths=1.2)
        self.full_pts = self.ax_full.scatter([], [], [], c="red", s=15)

        self.ax_full.add_collection3d(self.full_bones)
        self.ax_full.add_collection3d(self.full_x)
        self.ax_full.add_collection3d(self.full_y)
        self.ax_full.add_collection3d(self.full_z)

        # keypoint body
        self.kp_bones = Line3DCollection([], colors="black", linewidths=1.5, alpha=0.6)
        self.kp_x = Line3DCollection([], colors="red", linewidths=1.2)
        self.kp_y = Line3DCollection([], colors="green", linewidths=1.2)
        self.kp_z = Line3DCollection([], colors="blue", linewidths=1.2)
        self.kp_pts = self.ax_kp.scatter([], [], [], c="orange", s=15)
        # self.next_pts = self.ax_kp.scatter([], [], [], c="red", s=15)
        # self.kp_predict_pts = self.ax_kp.scatter([], [], [], c="green", s=15)

        self.ax_kp.add_collection3d(self.kp_bones)
        self.ax_kp.add_collection3d(self.kp_x)
        self.ax_kp.add_collection3d(self.kp_y)
        self.ax_kp.add_collection3d(self.kp_z)

        self._set_rotation_visible(True)

    def _set_rotation_visible(self, visible: bool):
        for col in [self.full_x, self.full_y, self.full_z, self.kp_x, self.kp_y, self.kp_z]:
            col.set_visible(visible)

    def _set_limits_and_title(self, ax, title: str, center: np.ndarray):
        ax.set_title(title)
        ax.set_xlim([center[0] - 0.8, center[0] + 0.8])
        ax.set_ylim([center[1] - 0.8, center[1] + 0.8])
        ax.set_zlim([0.0, 1.8])

    def _update_bones(self, bones_collection: Line3DCollection, pose_xyz: np.ndarray, pairs: List[tuple]):
        if len(pairs) == 0:
            bones_collection.set_segments([])
            return
        i_idx = np.array([i for i, _ in pairs], dtype=np.int64)
        p_idx = np.array([p for _, p in pairs], dtype=np.int64)
        segs = np.stack([pose_xyz[i_idx], pose_xyz[p_idx]], axis=1)  # (nb,2,3)
        bones_collection.set_segments(segs)

    def _update_points(self, scatter, pose_xyz: np.ndarray):
        scatter._offsets3d = (pose_xyz[:, 0], pose_xyz[:, 1], pose_xyz[:, 2])

    def _update_rotations(self, xcol, ycol, zcol, pose_xyz: np.ndarray, quat_wxyz: np.ndarray, scale: float = 0.08):
        quat_xyzw = quat_wxyz[:, [1, 2, 3, 0]]
        rot_mats = R.from_quat(quat_xyzw).as_matrix()  # (N,3,3)

        x_ends = pose_xyz + scale * rot_mats[:, :, 0]
        y_ends = pose_xyz + scale * rot_mats[:, :, 1]
        z_ends = pose_xyz + scale * rot_mats[:, :, 2]

        xcol.set_segments(np.stack([pose_xyz, x_ends], axis=1))
        ycol.set_segments(np.stack([pose_xyz, y_ends], axis=1))
        zcol.set_segments(np.stack([pose_xyz, z_ends], axis=1))

    def _redraw(self, force_draw: bool):
        if force_draw:
            self.fig.canvas.draw()
        else:
            self.fig.canvas.draw_idle()

        try:
            self.fig.canvas.flush_events()
        except Exception:
            plt.pause(0)

    def _render(self, frame_idx: int, force_draw: bool = False):
        if self.xyz is None or self.wxyz is None or self.curr_key is None or self.num_frames <= 0:
            self._redraw(force_draw=True)
            return

        frame_idx %= self.num_frames

        xyz = self.xyz[frame_idx]      # (J,3)
        # next_xyz = self.xyz[frame_idx + 1]
        wxyz = self.wxyz[frame_idx]    # (J,4)

        center = np.mean(xyz, axis=0)
        title = (
            f"[{self.curr_data_id+1}/{self.data_len}] {self.curr_key}\n"
            f"Full | Frame {frame_idx}/{self.num_frames}\n"
            f"fps={self.play_fps:.2f}\n"
            f"rotation={'on' if self.show_rotation else 'off'}"
        )
        self._set_limits_and_title(self.ax_full, title, center)
        self._update_bones(self.full_bones, xyz, self.body_bone_pairs)
        self._update_points(self.full_pts, xyz)

        kp_xyz = xyz[self.keypoints_indices]
        # kp_next_xyz = next_xyz[self.keypoints_indices]
        # kp_predict = add_bounded_noise(kp_xyz, min_mag=0.010, max_mag=0.020, seed=42)
        kp_wxyz = wxyz[self.keypoints_indices]
        kp_center = np.mean(kp_xyz, axis=0)
        kp_title = (
            f"[{self.curr_data_id+1}/{self.data_len}] {self.curr_key} \n"
            f"Keypoints | Frame {frame_idx}/{self.num_frames}\n"
            f"fps={self.play_fps:.2f}\n"
            f"rotation={'on' if self.show_rotation else 'off'}"
        )
        self._set_limits_and_title(self.ax_kp, kp_title, kp_center)
        self._update_bones(self.kp_bones, kp_xyz, self.kp_bone_pairs)
        self._update_points(self.kp_pts, kp_xyz)
        # self._update_points(self.next_pts, kp_next_xyz)
        # self._update_points(self.kp_predict_pts, kp_predict)

        self._set_rotation_visible(self.show_rotation)
        if self.show_rotation:
            self._update_rotations(self.full_x, self.full_y, self.full_z, xyz, wxyz)
            self._update_rotations(self.kp_x, self.kp_y, self.kp_z, kp_xyz, kp_wxyz)

        self._redraw(force_draw=force_draw)

    def save_gif(self, output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)
        for data_id, key in enumerate(self.keys):
            self.curr_data_id = data_id
            self._load_current_data()
            frames = []
            for i in range(self.num_frames):
                self._render(frame_idx=i, force_draw=True)
                self.fig.canvas.draw()
                buf = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
                w, h = self.fig.canvas.get_width_height()
                frames.append(buf.reshape(h, w, 3))
            safe_name = key.replace("/", "_").replace("\\", "_")
            out_path = os.path.join(output_dir, f"{safe_name}.gif")
            imageio.mimsave(out_path, frames, duration=1.0 / 2 * self.play_fps, loop=0)
            print(f"Saved: {out_path}  ({len(frames)} frames)")
        plt.close(self.fig)

    def run(self):
        if self.data_len == 0:
            raise ValueError("data is empty")

        plt.ion()
        try:
            plt.show(block=False)
        except Exception:
            pass

        next_tick = time.perf_counter()
        dropped_frames = 0

        while not self.is_closed:
            if self._switch_request != 0:
                self.curr_data_id = (self.curr_data_id + self._switch_request) % self.data_len
                self._switch_request = 0
                self.is_paused = False
                self._load_current_data()
                self.current_frame = 0
                self._render(0, force_draw=True)

                next_tick = time.perf_counter() + (1.0 / self.play_fps)
                continue

            if self.is_paused:
                self._redraw(force_draw=False)
                time.sleep(0.01)
                next_tick = time.perf_counter() + (1.0 / self.play_fps)
                continue

            if self.num_frames <= 0:
                self._redraw(force_draw=False)
                time.sleep(0.01)
                next_tick = time.perf_counter() + (1.0 / self.play_fps)
                continue

            period = 1.0 / max(self.play_fps, 1e-6)
            now = time.perf_counter()

            if now < next_tick:
                time.sleep(next_tick - now)
                now = time.perf_counter()

            if now - next_tick > period:
                behind = now - next_tick
                skip = int(behind // period)
                if skip > 0:
                    self.current_frame = (self.current_frame + skip) % self.num_frames
                    dropped_frames += skip
                    next_tick += skip * period
            self._render(self.current_frame, force_draw=False)
            self.current_frame = (self.current_frame + 1) % self.num_frames

            next_tick += period

        plt.ioff()

class RobotVisualizer:
    def __init__(self, data, humanoid_xml, dt=1/50):
        self.humanoid_xml = humanoid_xml
        self.dt = dt
        self.motion_data = data
        self.motion_keys = list(self.motion_data.keys())
        self.motion_id = 0
        self.time_step = 0
        self.paused = False
        self.request_close = False
        self.model = mujoco.MjModel.from_xml_path(humanoid_xml)
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = dt

    def add_visual_sphere(self, scene, position, radius, rgba):

        if scene.ngeom >= scene.maxgeom:
            return
        
        geom = scene.geoms[scene.ngeom]

        mujoco.mjv_initGeom(
            geom,
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=np.array([radius, 0, 0]),
            pos=position.astype(np.float32),
            mat=np.eye(3).flatten(),
            rgba=rgba.astype(np.float32)
        )
        scene.ngeom += 1
        
    def add_visual_capsule(self, scene, point1, point2, radius, rgba):
        if scene.ngeom >= scene.maxgeom:
            return
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            geom,
            mujoco.mjtGeom.mjGEOM_CAPSULE,
            np.zeros(3),
            np.zeros(3),
            np.zeros(9),
            rgba.astype(np.float32)
        )
        mujoco.mjv_makeConnector(
            geom,
            mujoco.mjtGeom.mjGEOM_CAPSULE,
            radius,
            *point1, *point2
        )
        scene.ngeom += 1

    def key_callback(self, keycode):
        key = chr(keycode)
        if key == "R":
            print("Resetting time step.")
            self.time_step = 0
        elif key == " ":
            self.paused = not self.paused
            print("Paused" if self.paused else "Unpaused")
        elif key == "N":
            self.motion_id = (self.motion_id + 1) % len(self.motion_keys)
            self.time_step = 0
            print(f"Switched to: {self.motion_keys[self.motion_id]}")
        elif key == "U":
            self.motion_id = (self.motion_id - 1) % len(self.motion_keys)
            print(f"Switched to: {self.motion_keys[self.motion_id]}")
        elif key == "C":
            print("Closing MuJoCo viewer.")
            self.request_close = True
        elif key == "A":
            curr_key = self.motion_keys[self.motion_id]
            print("Abandon:", curr_key)
            os.makedirs("tmp/relabel", exist_ok=True)
            with open("tmp/relabel/abandoned.txt", "a") as f:
                f.write(curr_key + "\n")
        elif key == "S":
            curr_key = self.motion_keys[self.motion_id]
            print(f"Saving: {curr_key}")
            os.makedirs("tmp/relabel", exist_ok=True)
            with open("tmp/relabel/selected.txt", "a") as f:
                f.write(curr_key + "\n")
        # else:
        #     print("Unmapped key:", key)

    def update_viewer(self, viewer):
        curr_key = self.motion_keys[self.motion_id]
        curr_motion = self.motion_data[curr_key]
        curr_time = int(self.time_step / self.dt) % curr_motion['reset_joint_pos'].shape[0]

        self.data.qpos[:3] = curr_motion["reset_root_trans"][curr_time]
        self.data.qpos[3:7] = curr_motion["reset_root_rot"][curr_time]
        self.data.qpos[7:] = curr_motion["reset_joint_pos"][curr_time]

        mujoco.mj_forward(self.model, self.data)
        joints = curr_motion['body_pos'][curr_time]
        # print(f"joints shape: {joints.shape}")
        for i, pos in enumerate(joints):
            viewer.user_scn.geoms[i].pos = pos

    def run(self):
        with mujoco.viewer.launch_passive(self.model, self.data, key_callback=self.key_callback) as viewer:
            for i in range(23):
                radius = 0.05
                if i in (9, 10, 11):
                    radius /= 1.5
                self.add_visual_capsule(viewer.user_scn, np.zeros(3), np.array([0.001, 0, 0]), radius, np.fromstring(HUMAN_BODY_LINKS_COLOR[i], dtype=float, sep=" "))
            if self.motion_id == 0:
                    print(f"start from motion_id = {self.motion_id} : {self.motion_keys[self.motion_id]}")
            while viewer.is_running():
                if self.request_close:
                    viewer.close()
                step_start = time.time()
                self.update_viewer(viewer)
                if not self.paused:
                    self.time_step += self.dt
                viewer.sync()
                elapsed = time.time() - step_start
                if elapsed < self.dt:
                    time.sleep(self.dt - elapsed)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize G1 motion data from a single file or a directory."
    )
    parser.add_argument(
        "--motion-path",
        default="/liujinxin/zhaowei/G1_MOTION/data/G1_motion_data/amass_142_18_poses_Anim_2209.pkl",
        help="Path to a .pkl/.json file or a directory containing motion files. Default: G1_motion_data",
    )
    parser.add_argument(
        "--type",
        choices=["pkl", "json"],
        default=None,
        help="Motion file type. Inferred from file extension when motion_path is a file.",
    )
    parser.add_argument(
        "--bad-dir",
        default="Bad_motion_data",
        help="Directory used when pressing b to save the current motion. Default: Bad_motion_data",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=50.0,
        help="Playback FPS for the matplotlib visualizer. Default: 50",
    )
    parser.add_argument(
        "--viewer",
        choices=["motion", "robot"],
        default="motion",
        help="Visualizer backend to run. Default: motion",
    )
    parser.add_argument(
        "--humanoid-xml",
        default="resources/robots/g1/g1_skeleton.xml",
        help="MuJoCo XML path used by --viewer robot.",
    )
    parser.add_argument(
        "--output-dir",
        default="output_gifs",
        help="Directory to save GIF files. Default: output_gifs",
    )
    return parser.parse_args()


def infer_file_type(motion_path: str, explicit_type: Optional[str]) -> str:
    if explicit_type is not None:
        return explicit_type
    ext = os.path.splitext(motion_path)[1].lower()
    if ext == ".json":
        return "json"
    return "pkl"


def main():
    args = parse_args()
    file_type = infer_file_type(args.motion_path, args.type)

    if not os.path.exists(args.motion_path):
        raise FileNotFoundError(f"motion_path does not exist: {args.motion_path}")

    data = Data_Loader.load_motion_data(args.motion_path, file_type)
    if len(data) == 0:
        raise ValueError(f"No {file_type} motion data found in: {args.motion_path}")

    if args.viewer == "motion":
        motion_vizer = MotionVisualizer(data, args.bad_dir, fps=args.fps)
        motion_vizer.save_gif(args.output_dir)
    else:
        robot_vizer = RobotVisualizer(data, args.humanoid_xml, dt=1.0 / args.fps)
        robot_vizer.run()

if __name__ == "__main__":
    main() 
