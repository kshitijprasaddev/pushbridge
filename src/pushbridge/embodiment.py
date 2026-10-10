"""Contact-space retargeting to a compact, speed-limited redundant arm.

The policy remains in two-dimensional task space. A constrained damped
least-squares solver converts tool targets to three joint servo commands.
Only reset assigns the object's pose; all subsequent motion uses contacts.
"""
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from .sim import DT, HEIGHT, ORIGIN, WIDTH, PushEnv, clip_action
import mujoco


COMPACT_SPEC = {
    'name': 'Compact 3R', 'link_lengths_m': [.24, .17, .13],
    'joint_ranges_rad': [[-1.3, 1.3], [.05, 2.75], [-1.6, 1.6]],
    'max_command_joint_speed_rad_s': .25, 'tip_radius_m': .003,
    'puck_radius_m': .012, 'puck_mass_kg': .05,
    'servo_kp': 2200, 'servo_kv': 30,
    'solver': 'Damped least squares with joint bounds, command slew limits and a null-space posture preference',
}


def compact_xml():
    root = ET.parse(Path(__file__).parent/'assets/push.xml').getroot()
    root.set('model', 'PushBridge compact redundant arm')
    shoulder = root.find(".//body[@name='shoulder']")
    elbow = shoulder.find("body[@name='elbow']")
    tool = elbow.find("body[@name='tool']")
    shoulder.find("joint").set('range', '-1.3 1.3')
    elbow.find("joint").set('range', '.05 2.75')
    shoulder.find("geom[@type='capsule']").set('fromto', '0 0 0 .24 0 0')
    elbow.set('pos', '.24 0 0')
    elbow.find("geom[@type='capsule']").set('fromto', '0 0 0 .17 0 0')
    elbow.remove(tool)
    wrist = ET.SubElement(elbow, 'body', name='wrist', pos='.17 0 0')
    ET.SubElement(wrist, 'joint', name='wrist', type='hinge', axis='0 0 1', range='-1.6 1.6')
    ET.SubElement(wrist, 'geom', type='sphere', size='.020', mass='.06', rgba='.12 .25 .32 1', contype='0', conaffinity='0')
    ET.SubElement(wrist, 'geom', type='capsule', **{'fromto': '0 0 0 .13 0 0', 'size': '.012', 'mass': '.10', 'rgba': '.20 .68 .58 1', 'contype': '0', 'conaffinity': '0'})
    tool.set('pos', '.13 0 0')
    tool.find("geom[@name='tip_geom']").set('size', '.003')
    wrist.append(tool)
    root.find(".//geom[@name='puck_geom']").set('mass', '.05')
    actuators = root.find('actuator')
    for name in ['shoulder', 'elbow', 'wrist']:
        servo = actuators.find(f"position[@joint='{name}']")
        if servo is None:
            servo = ET.SubElement(actuators, 'position', name='wrist_servo', joint='wrist', forcerange='-25 25')
        servo.set('kp', '2200')
        servo.set('kv', '30')
        lo, hi = COMPACT_SPEC['joint_ranges_rad'][['shoulder', 'elbow', 'wrist'].index(name)]
        servo.set('ctrlrange', f'{lo} {hi}')
        servo.set('ctrllimited', 'true')
    return ET.tostring(root, encoding='unicode')


class CompactPushEnv(PushEnv):
    lengths = np.array(COMPACT_SPEC['link_lengths_m'])
    limits = np.array(COMPACT_SPEC['joint_ranges_rad'])
    home = np.array([-.7, 1.7, -.9])

    def __init__(self, friction=.25):
        self.model = mujoco.MjModel.from_xml_string(compact_xml())
        self.data = mujoco.MjData(self.model)
        for name in ['floor', 'puck_geom']:
            self.model.geom(name).friction[0] = friction
        self.tip_id = self.model.site('tip').id
        self.puck_addr = int(self.model.joint('puck').qposadr[0])
        self.renderer = None
        self.command = self.home.copy()
        self.adapter_diagnostics = {}

    @classmethod
    def fk_jacobian(cls, q):
        angle = np.cumsum(q)
        sx, sy = cls.lengths*np.cos(angle), cls.lengths*np.sin(angle)
        position = np.array([sx.sum(), sy.sum()])
        jacobian = np.array([[-sy[j:].sum() for j in range(3)], [sx[j:].sum() for j in range(3)]])
        return position, jacobian

    @classmethod
    def solve(cls, local_xy, q, iterations=20):
        target = np.asarray(local_xy)+ORIGIN
        q = np.asarray(q).copy()
        for _ in range(iterations):
            pos, jac = cls.fk_jacobian(q)
            error = target-pos
            if np.linalg.norm(error) < 1e-5:
                break
            inverse = jac.T@np.linalg.inv(jac@jac.T+np.eye(2)*1e-5)
            null = np.eye(3)-inverse@jac
            change = inverse@error + .008*null@(cls.home-q)
            q = np.clip(q+np.clip(change, -.18, .18), cls.limits[:, 0]+.002, cls.limits[:, 1]-.002)
        return q

    def reset(self, puck=(.10, .105), tool=(.06, .105), goal=(.23, .105)):
        mujoco.mj_resetData(self.model, self.data)
        q = self.solve(tool, self.home, 100)
        residual = np.linalg.norm(self.fk_jacobian(q)[0]-(np.asarray(tool)+ORIGIN))
        if residual > .0005:
            raise ValueError(f'Compact arm cannot reach reset tool position: {residual*1000:.2f} mm residual')
        self.data.qpos[:3] = q
        self.data.ctrl[:] = q
        self.command = q.copy()
        self.data.qpos[self.puck_addr:self.puck_addr+3] = [*(np.asarray(puck)+ORIGIN), .0091]
        self.data.qpos[self.puck_addr+3:self.puck_addr+7] = [1, 0, 0, 0]
        self.model.site('goal').pos[:2] = np.asarray(goal)+ORIGIN
        self.goal = np.asarray(goal).copy()
        mujoco.mj_forward(self.model, self.data)
        for _ in range(100):
            mujoco.mj_step(self.model, self.data)
        self.adapter_diagnostics = {'control_steps': 0, 'slew_limited_steps': 0,
                                    'max_command_joint_speed_rad_s': 0., 'sum_tracking_residual_m': 0.}
        return self.state()

    def step(self, action):
        before = self.state()[:2]
        target = np.clip(before+clip_action(action), [.012, .012], [WIDTH-.012, HEIGHT-.012])
        proposed = self.solve(target, self.data.qpos[:3], 12)
        limit = COMPACT_SPEC['max_command_joint_speed_rad_s']*DT
        delta = proposed-self.command
        limited = np.clip(delta, -limit, limit)
        q0 = self.command.copy()
        self.command = np.clip(q0+limited, self.limits[:, 0]+.002, self.limits[:, 1]-.002)
        for i in range(50):
            self.data.ctrl[:] = q0+(self.command-q0)*(i+1)/50
            mujoco.mj_step(self.model, self.data)
        state = self.state()
        if not np.isfinite(state).all():
            raise RuntimeError('Compact simulation diverged')
        d = self.adapter_diagnostics
        d['control_steps'] += 1
        d['slew_limited_steps'] += int(np.any(np.abs(delta) > limit+1e-9))
        d['max_command_joint_speed_rad_s'] = max(d['max_command_joint_speed_rad_s'], float(np.abs(limited).max()/DT))
        d['sum_tracking_residual_m'] += float(np.linalg.norm(target-state[:2]))
        return state


CONTACT_SCALE = (.012+.008)/(.012+COMPACT_SPEC['tip_radius_m'])


def canonical_state(state):
    state = np.asarray(state).copy()
    state[:2] = state[2:] + CONTACT_SCALE*(state[:2]-state[2:])
    return state


class ContactAdapter:
    """Normalise finger/tool clearance and undo its action scaling.

    The hand-trained/source-arm policy operates at a 20 mm centre-to-centre
    contact radius. The compact tool contacts at 15 mm. This transformation
    handles tool geometry; it does not correct friction, mass or tracking lag.
    """
    def __init__(self, policy):
        self.policy = policy

    def act(self, state, goal):
        return clip_action(self.policy.act(canonical_state(state), goal)/CONTACT_SCALE)


class CompactGeometry:
    """Embodiment-specific classical control: slow pushes with lateral feedback.

    This second baseline accounts for the narrow tip and servo lag. Constants
    come from known tip/object geometry, not fitted demonstrations.
    """
    def act(self, state, goal):
        tool, puck = state[:2], state[2:]
        direction = goal-puck
        direction /= max(np.linalg.norm(direction), 1e-8)
        relative = tool-puck
        along = float(relative@direction)
        lateral = relative-direction*along
        if along > -.010:
            side = np.array([-direction[1], direction[0]])
            sign = 1 if relative@side >= 0 else -1
            desired = puck-direction*.020+side*sign*.025
            return clip_action(desired-tool, .004)
        if np.linalg.norm(lateral) > .0015 or along < -.019:
            return clip_action(puck-direction*.017-tool, .004)
        return clip_action(direction*.004-.8*lateral, .004)
