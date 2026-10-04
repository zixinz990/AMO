"""Headless port of the AMO controller in play_amo.py (HumanoidEnv without the GLFW viewer).

Observation construction, adapter call, action processing, arm blending, gait-phase logic
and the PD/torque loop are kept identical to play_amo.py; only the keyboard-driven
`viewer.commands` vector is replaced by `self.commands`, which a script writes to.

commands layout (same as play_amo.py):
    [0] vx            [1] target yaw (absolute heading)   [2] vy
    [3] height offset (torso height = 0.75 + commands[3])
    [4] torso yaw     [5] torso pitch    [6] torso roll    [7] random-arm toggle (T key)
"""

from collections import deque
from pathlib import Path

import mujoco
import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent

# The joint order of the 23 actuated DOFs in g1.xml
DOF_NAMES = ["left_hip_pitch", "left_hip_roll", "left_hip_yaw", "left_knee", "left_ankle_pitch", "left_ankle_roll",
             "right_hip_pitch", "right_hip_roll", "right_hip_yaw", "right_knee", "right_ankle_pitch", "right_ankle_roll",
             "waist_yaw", "waist_roll", "waist_pitch",
             "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw", "left_elbow",
             "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw", "right_elbow"]
STIFFNESS = np.array([
    150, 150, 150, 300, 80, 20,
    150, 150, 150, 300, 80, 20,
    400, 400, 400,
    80, 80, 40, 60,
    80, 80, 40, 60,
])
DAMPING = np.array([
    2, 2, 2, 4, 2, 1,
    2, 2, 2, 4, 2, 1,
    15, 15, 15,
    2, 2, 1, 1,
    2, 2, 1, 1,
])
DEFAULT_DOF_POS = np.array([
    -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
    -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
    0.0, 0.0, 0.0,
    0.5, 0.0, 0.2, 0.3,
    0.5, 0.0, -0.2, 0.3,
])
TORQUE_LIMITS = np.array([
    88, 139, 88, 139, 50, 50,
    88, 139, 88, 139, 50, 50,
    88, 50, 50,
    25, 25, 25, 25,
    25, 25, 25, 25,
])
SIM_DT = 0.002
SIM_DECIMATION = 10
CONTROL_DT = SIM_DT * SIM_DECIMATION
BASE_HEIGHT = 0.75


def quatToEuler(quat):
    eulerVec = np.zeros(3)
    qw = quat[0]
    qx = quat[1]
    qy = quat[2]
    qz = quat[3]
    # roll (x-axis rotation)
    sinr_cosp = 2 * (qw * qx + qy * qz)
    cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
    eulerVec[0] = np.arctan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2 * (qw * qy - qz * qx)
    if np.abs(sinp) >= 1:
        eulerVec[1] = np.copysign(np.pi / 2, sinp)  # use 90 degrees if out of range
    else:
        eulerVec[1] = np.arcsin(sinp)

    # yaw (z-axis rotation)
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    eulerVec[2] = np.arctan2(siny_cosp, cosy_cosp)

    return eulerVec


class AMOController:
    """play_amo.HumanoidEnv with the viewer removed; `step()` advances one physics step."""

    def __init__(self, model, data, device="cpu", seed=0):
        self.model, self.data = model, data
        self.device = device
        self.rng = np.random.default_rng(seed)

        self.stiffness = STIFFNESS
        self.damping = DAMPING
        self.num_actions = 15
        self.num_dofs = 23
        self.default_dof_pos = DEFAULT_DOF_POS
        self.torque_limits = TORQUE_LIMITS
        self.arm_dof_lower_range = -0.4 * np.ones(8)
        self.arm_dof_upper_range = 0.4 * np.ones(8)
        self.sim_dt = SIM_DT
        self.sim_decimation = SIM_DECIMATION
        self.control_dt = CONTROL_DT

        self.commands = np.zeros(8, dtype=np.float32)

        self.last_action = np.zeros(self.num_actions, dtype=np.float32)
        self.action_scale = 0.25
        self.arm_action = self.default_dof_pos[15:]
        self.prev_arm_action = self.default_dof_pos[15:]
        self.arm_blend = 0.0
        self.toggle_arm = False

        self.scales_ang_vel = 0.25
        self.scales_dof_vel = 0.05

        self.nj = 23
        self.n_priv = 3
        self.n_proprio = 3 + 2 + 2 + 23 * 3 + 2 + 15
        self.history_len = 10
        self.extra_history_len = 25
        self._n_demo_dof = 8

        self.dof_pos = np.zeros(self.nj, dtype=np.float32)
        self.dof_vel = np.zeros(self.nj, dtype=np.float32)
        self.quat = np.zeros(4, dtype=np.float32)
        self.ang_vel = np.zeros(3, dtype=np.float32)
        self.last_action = np.zeros(self.nj)

        self.demo_obs_template = np.zeros((8 + 3 + 3 + 3, ))
        self.demo_obs_template[:self._n_demo_dof] = self.default_dof_pos[15:]
        self.demo_obs_template[self._n_demo_dof + 6:self._n_demo_dof + 9] = 0.75

        self.target_yaw = 0.0

        self._in_place_stand_flag = True
        self.gait_cycle = np.array([0.25, 0.25])
        self.gait_freq = 1.3

        self.proprio_history_buf = deque(maxlen=self.history_len)
        self.extra_history_buf = deque(maxlen=self.extra_history_len)
        for i in range(self.history_len):
            self.proprio_history_buf.append(np.zeros(self.n_proprio))
        for i in range(self.extra_history_len):
            self.extra_history_buf.append(np.zeros(self.n_proprio))

        self.policy_jit = torch.jit.load(str(REPO / "amo_jit.pt"), map_location=self.device)

        self.adapter = torch.jit.load(str(REPO / "adapter_jit.pt"), map_location=self.device)
        self.adapter.eval()
        for param in self.adapter.parameters():
            param.requires_grad = False

        norm_stats = torch.load(str(REPO / "adapter_norm_stats.pt"))
        self.input_mean = torch.tensor(norm_stats['input_mean'], device=self.device, dtype=torch.float32)
        self.input_std = torch.tensor(norm_stats['input_std'], device=self.device, dtype=torch.float32)
        self.output_mean = torch.tensor(norm_stats['output_mean'], device=self.device, dtype=torch.float32)
        self.output_std = torch.tensor(norm_stats['output_std'], device=self.device, dtype=torch.float32)

        self.adapter_input = torch.zeros((1, 8 + 4), device=self.device, dtype=torch.float32)
        self.adapter_output = torch.zeros((1, 15), device=self.device, dtype=torch.float32)

        self.i = 0
        self.pd_target = None

    # ------------------------------------------------------------------ scripted arm input
    def set_arm_target(self, arm_q):
        """Start a linear blend from the measured arm pose to `arm_q` (8 DOFs).

        Uses the same mechanism (prev_arm_action / arm_action / arm_blend reset) that
        play_amo.py uses when the T toggle samples a new random arm target.
        """
        self.arm_blend = 0
        self.prev_arm_action = self.dof_pos[15:].copy()
        self.arm_action = np.asarray(arm_q, dtype=np.float64).copy()

    # ------------------------------------------------------------------ play_amo.py logic
    def extract_data(self):
        self.dof_pos = self.data.qpos.astype(np.float32)[-self.num_dofs:]
        self.dof_vel = self.data.qvel.astype(np.float32)[-self.num_dofs:]
        self.quat = self.data.sensor('orientation').data.astype(np.float32)
        self.ang_vel = self.data.sensor('angular-velocity').data.astype(np.float32)

    def get_observation(self):
        rpy = quatToEuler(self.quat)

        self.target_yaw = self.commands[1]
        dyaw = rpy[2] - self.target_yaw
        dyaw = np.remainder(dyaw + np.pi, 2 * np.pi) - np.pi
        if self._in_place_stand_flag:
            dyaw = 0.0

        obs_dof_vel = self.dof_vel.copy()
        obs_dof_vel[[4, 5, 10, 11, 13, 14]] = 0.0

        gait_obs = np.sin(self.gait_cycle * 2 * np.pi)

        self.adapter_input = np.concatenate([np.zeros(4), self.dof_pos[15:]])

        self.adapter_input[0] = 0.75 + self.commands[3]
        self.adapter_input[1] = self.commands[4]
        self.adapter_input[2] = self.commands[5]
        self.adapter_input[3] = self.commands[6]

        self.adapter_input = torch.tensor(self.adapter_input).to(self.device, dtype=torch.float32).unsqueeze(0)

        self.adapter_input = (self.adapter_input - self.input_mean) / (self.input_std + 1e-8)
        self.adapter_output = self.adapter(self.adapter_input.view(1, -1))
        self.adapter_output = self.adapter_output * self.output_std + self.output_mean

        obs_prop = np.concatenate([
            self.ang_vel * self.scales_ang_vel,
            rpy[:2],
            (np.sin(dyaw),
             np.cos(dyaw)),
            (self.dof_pos - self.default_dof_pos),
            self.dof_vel * self.scales_dof_vel,
            self.last_action,
            gait_obs,
            self.adapter_output.cpu().numpy().squeeze(),
        ])

        obs_priv = np.zeros((self.n_priv, ))
        obs_hist = np.array(self.proprio_history_buf).flatten()

        obs_demo = self.demo_obs_template.copy()
        obs_demo[:self._n_demo_dof] = self.dof_pos[15:]
        obs_demo[self._n_demo_dof] = self.commands[0]
        obs_demo[self._n_demo_dof + 1] = self.commands[2]
        self._in_place_stand_flag = np.abs(self.commands[0]) < 0.1
        obs_demo[self._n_demo_dof + 3] = self.commands[4]
        obs_demo[self._n_demo_dof + 4] = self.commands[5]
        obs_demo[self._n_demo_dof + 5] = self.commands[6]
        obs_demo[self._n_demo_dof + 6:self._n_demo_dof + 9] = 0.75 + self.commands[3]

        self.proprio_history_buf.append(obs_prop)
        self.extra_history_buf.append(obs_prop)

        return np.concatenate((obs_prop, obs_demo, obs_priv, obs_hist))

    def policy_tick(self):
        """Body of the `if i % self.sim_decimation == 0` branch of HumanoidEnv.run()."""
        i = self.i
        obs = self.get_observation()

        obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            extra_hist = torch.tensor(np.array(self.extra_history_buf).flatten().copy(), dtype=torch.float).view(1, -1).to(self.device)
            raw_action = self.policy_jit(obs_tensor, extra_hist).cpu().numpy().squeeze()

        raw_action = np.clip(raw_action, -40., 40.)
        self.last_action = np.concatenate([raw_action.copy(), (self.dof_pos - self.default_dof_pos)[15:] / self.action_scale])
        scaled_actions = raw_action * self.action_scale

        if i % 300 == 0 and i > 0 and self.commands[7]:
            self.arm_blend = 0
            self.prev_arm_action = self.dof_pos[15:].copy()
            self.arm_action = self.rng.uniform(0, 1, 8) * (self.arm_dof_upper_range - self.arm_dof_lower_range) + self.arm_dof_lower_range
            self.toggle_arm = True
        elif not self.commands[7]:
            if self.toggle_arm:
                self.toggle_arm = False
                self.arm_blend = 0
                self.prev_arm_action = self.dof_pos[15:].copy()
                self.arm_action = self.default_dof_pos[15:]
        pd_target = np.concatenate([scaled_actions, np.zeros(8)]) + self.default_dof_pos
        pd_target[15:] = (1 - self.arm_blend) * self.prev_arm_action + self.arm_blend * self.arm_action
        self.arm_blend = min(1.0, self.arm_blend + 0.01)

        self.gait_cycle = np.remainder(self.gait_cycle + self.control_dt * self.gait_freq, 1.0)
        if self._in_place_stand_flag and ((np.abs(self.gait_cycle[0] - 0.25) < 0.05) or (np.abs(self.gait_cycle[1] - 0.25) < 0.05)):
            self.gait_cycle = np.array([0.25, 0.25])
        if (not self._in_place_stand_flag) and ((np.abs(self.gait_cycle[0] - 0.25) < 0.05) and (np.abs(self.gait_cycle[1] - 0.25) < 0.05)):
            self.gait_cycle = np.array([0.25, 0.75])
        self.pd_target = pd_target

    def step(self):
        """One iteration of HumanoidEnv.run(): (policy every 10th step) + PD torque + mj_step."""
        self.extract_data()
        if self.i % self.sim_decimation == 0:
            self.policy_tick()

        torque = (self.pd_target - self.dof_pos) * self.stiffness - self.dof_vel * self.damping
        torque = np.clip(torque, -self.torque_limits, self.torque_limits)

        self.data.ctrl = torque

        mujoco.mj_step(self.model, self.data)
        self.i += 1


def make_model(path=REPO / "g1.xml", skybox=False):
    """Load g1.xml and initialise it exactly like HumanoidEnv.__init__.

    skybox=True adds the gradient skybox of the SONIC / FALCON demo scenes (g1.xml has none).
    It is a texture only, so the dynamics are bit-identical to loading g1.xml directly.
    """
    if skybox:
        spec = mujoco.MjSpec()
        spec.from_file(str(path))
        tex = spec.add_texture()
        tex.type = mujoco.mjtTexture.mjTEXTURE_SKYBOX
        tex.builtin = mujoco.mjtBuiltin.mjBUILTIN_GRADIENT
        tex.rgb1, tex.rgb2 = [0.3, 0.5, 0.7], [0.0, 0.0, 0.0]
        tex.width, tex.height, tex.nchannel = 512, 3072, 3
        model = spec.compile()
    else:
        model = mujoco.MjModel.from_xml_path(str(path))
    model.opt.timestep = SIM_DT
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_step(model, data)
    return model, data
