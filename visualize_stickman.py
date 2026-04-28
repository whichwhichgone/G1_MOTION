import time
import joblib
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection

# 节点索引 (来自 HUMAN_BODY_LINKS)
# idx: 0=pelvis, 3=left_foot, 7=right_foot, 11=spine2, 15=left_wrist, 19=right_wrist
KEYPOINT_INDICES = [0, 3, 7, 11, 15, 19]
# 局部索引映射
PELVIS, LEFT_FOOT, RIGHT_FOOT, SPINE2, LEFT_WRIST, RIGHT_WRIST = range(6)

# 骨骼连线 (局部索引)
BONE_PAIRS = [
    (PELVIS, SPINE2),       # 躯干
    (SPINE2, LEFT_WRIST),   # 左臂
    (SPINE2, RIGHT_WRIST),  # 右臂
    (PELVIS, LEFT_FOOT),    # 左腿
    (PELVIS, RIGHT_FOOT),   # 右腿
]

POINT_COLORS = ["red", "purple", "orange", "gold", "blue", "green"]
POINT_LABELS = ["pelvis", "L_foot", "R_foot", "spine2", "L_wrist", "R_wrist"]


def load_stickman(path: str) -> np.ndarray:
    data = joblib.load(path)
    body_pos = np.asarray(data["body_pos"])            # (T, 23, 3)
    kp = body_pos[:, KEYPOINT_INDICES, :]              # (T, 6, 3)
    return kp.reshape(len(kp), -1)                     # (T, 18)


class StickmanVisualizer:

    def __init__(self, kp_xyz: np.ndarray, title: str = "", fps: float = 50.0):
        self.kp_xyz = kp_xyz          # (T, 6, 3)
        self.num_frames = kp_xyz.shape[0]
        self.title = title
        self.play_fps = fps
        self.current_frame = 0
        self.is_paused = False
        self.is_closed = False

        self.fig = plt.figure(figsize=(8, 8))
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")

        self.bones = Line3DCollection([], colors="black", linewidths=2.0, alpha=0.8)
        self.ax.add_collection3d(self.bones)
        self.scatters = [
            self.ax.scatter([], [], [], c=c, s=60, label=l, zorder=5)
            for c, l in zip(POINT_COLORS, POINT_LABELS)
        ]
        self.ax.legend(loc="upper right", fontsize=8)

        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self.fig.canvas.mpl_connect("close_event", lambda e: setattr(self, "is_closed", True))

    def _on_key(self, event):
        k = (event.key or "").lower()
        if k == " ":
            self.is_paused = not self.is_paused
        elif k == "r":
            self.current_frame = 0
        elif k in [".", ">"]:
            self.is_paused = True
            self.current_frame = (self.current_frame + 1) % self.num_frames
            self._render(force_draw=True)
        elif k in [",", "<"]:
            self.is_paused = True
            self.current_frame = (self.current_frame - 1) % self.num_frames
            self._render(force_draw=True)
        elif k in ["q", "escape"]:
            self.is_closed = True
            plt.close(self.fig)

    def _render(self, force_draw=False):
        frame_idx = self.current_frame % self.num_frames
        xyz = self.kp_xyz[frame_idx]   # (5, 3)

        # 更新骨骼连线
        segs = np.stack([xyz[[i for i, _ in BONE_PAIRS]],
                         xyz[[j for _, j in BONE_PAIRS]]], axis=1)
        self.bones.set_segments(segs)

        # 更新各节点散点
        for idx, sc in enumerate(self.scatters):
            sc._offsets3d = ([xyz[idx, 0]], [xyz[idx, 1]], [xyz[idx, 2]])

        # 更新坐标范围
        center = np.mean(xyz, axis=0)
        self.ax.set_xlim([center[0] - 1.0, center[0] + 1.0])
        self.ax.set_ylim([center[1] - 1.0, center[1] + 1.0])
        self.ax.set_zlim([0.0, 2.0])
        self.ax.set_title(f"{self.title}\nFrame {frame_idx}/{self.num_frames}  fps={self.play_fps:.0f}\n"
                          f"[Space] pause  [R] reset  [,/.] step  [Q] quit")

        if force_draw:
            self.fig.canvas.draw()
        else:
            self.fig.canvas.draw_idle()
        try:
            self.fig.canvas.flush_events()
        except Exception:
            plt.pause(0)

    def run(self):
        plt.ion()
        plt.show(block=False)
        next_tick = time.perf_counter()
        period = 1.0 / self.play_fps

        while not self.is_closed:
            if self.is_paused:
                self._render(force_draw=False)
                time.sleep(0.01)
                next_tick = time.perf_counter() + period
                continue

            now = time.perf_counter()
            if now < next_tick:
                time.sleep(next_tick - now)

            self._render(force_draw=False)
            self.current_frame = (self.current_frame + 1) % self.num_frames
            next_tick += period

        plt.ioff()


if __name__ == "__main__":
    pkl_path = "data/G1_motion_data/amass_Trial_04_poses_01_Anim_3925.pkl"
    kp_xyz = load_stickman(pkl_path)
    kp_xyz = kp_xyz.reshape(-1, 6, 3)
    viz = StickmanVisualizer(kp_xyz, title=pkl_path, fps=50.0)
    viz.run()
