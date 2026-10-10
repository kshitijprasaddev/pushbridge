"""A physical pushing environment. Only initial reset sets puck pose directly."""
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco
import numpy as np

WIDTH, HEIGHT = 0.297, 0.210
ORIGIN = np.array([0.25, -0.105])
DT = 0.1
MAX_ACTION = 0.008


def clip_action(action, limit=MAX_ACTION):
    a = np.asarray(action, dtype=float)
    return a * min(1.0, limit / max(np.linalg.norm(a), 1e-10))


class PushEnv:
    def __init__(self, friction=0.25):
        self.model = mujoco.MjModel.from_xml_path(str(Path(__file__).parent / "assets/push.xml"))
        self.data = mujoco.MjData(self.model)
        self.model.geom("floor").friction[0] = friction
        self.model.geom("puck_geom").friction[0] = friction
        self.tip_id = self.model.site("tip").id
        self.puck_addr = int(self.model.joint("puck").qposadr[0])
        self.renderer = None

    @staticmethod
    def ik(local_xy):
        x, y = np.asarray(local_xy) + ORIGIN
        c = (x*x + y*y - 0.32**2 - 0.30**2) / (2 * 0.32 * 0.30)
        if abs(c) > 1:
            raise ValueError("Target outside the arm's reach")
        elbow = np.arccos(c)
        shoulder = np.arctan2(y, x) - np.arctan2(0.30*np.sin(elbow), 0.32+0.30*np.cos(elbow))
        return np.array([shoulder, elbow])

    def reset(self, puck=(0.10, 0.105), tool=(0.06, 0.105), goal=(0.23, 0.105)):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:2] = self.ik(tool)
        self.data.ctrl[:] = self.data.qpos[:2]
        self.data.qpos[self.puck_addr:self.puck_addr+3] = [*(np.asarray(puck)+ORIGIN), 0.0091]
        self.data.qpos[self.puck_addr+3:self.puck_addr+7] = [1, 0, 0, 0]
        self.model.site("goal").pos[:2] = np.asarray(goal) + ORIGIN
        self.goal = np.asarray(goal).copy()
        mujoco.mj_forward(self.model, self.data)
        for _ in range(100):
            mujoco.mj_step(self.model, self.data)
        return self.state()

    def state(self):
        return np.r_[self.data.site_xpos[self.tip_id, :2]-ORIGIN,
                     self.data.body("puck").xpos[:2]-ORIGIN].copy()

    def step(self, action):
        old = self.state()[:2]
        target = np.clip(old + clip_action(action), [0.012, 0.012], [WIDTH-0.012, HEIGHT-0.012])
        for i in range(50):
            waypoint = old + (target-old)*(i+1)/50
            self.data.ctrl[:] = self.ik(waypoint)
            mujoco.mj_step(self.model, self.data)
        state = self.state()
        if not np.all(np.isfinite(state)):
            raise RuntimeError("Simulation diverged")
        return state

    def render(self):
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, height=640, width=960)
        self.renderer.update_scene(self.data, camera="overview")
        return self.renderer.render().copy()

    def close(self):
        if self.renderer:
            self.renderer.close()
            self.renderer = None
