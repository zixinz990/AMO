"""Headless AMO sim2sim in MuJoCo with scripted locomotion + arm commands, rendered to mp4.

Physics and controller are those of play_amo.py (AMO's g1.xml, 500 Hz physics, 50 Hz policy,
PD torques with the same gains and torque limits, same initial keyframe); see amo_policy.py.
Only the keyboard input is replaced by a command script and the GLFW viewer by an offscreen
renderer. The command schedule is the one of the SONIC MuJoCo demo
(GR00T-WholeBodyControl/sonic_mujoco_test), mapped onto AMO's command interface:
    velocity -> commands[0] (vx), commands[2] (vy)
    turning  -> commands[1], the absolute target heading, integrated from the turn rate
    EE poses -> arm joint targets (AMO's arms are PD-tracked joint targets, blended in over 2 s)
--torso appends AMO-specific torso commands (height, yaw, pitch, roll) that SONIC does not have.

Usage (from the repo root):
    source amo_mujoco_test/cache_env.sh && source .venv/bin/activate
    python amo_mujoco_test/run_amo_mujoco.py --out amo_mujoco_test/amo_mujoco_demo.mp4
"""

import argparse
import json
import math
import os
from pathlib import Path
import time

os.environ.setdefault("MUJOCO_GL", "egl")

import cv2
import imageio.v2 as imageio
import mujoco
import numpy as np

from amo_policy import (
    BASE_HEIGHT,
    CONTROL_DT,
    DEFAULT_DOF_POS,
    SIM_DECIMATION,
    SIM_DT,
    AMOController,
    make_model,
    quatToEuler,
)

# Hand key points (palm centre) for the arm-target markers
HAND_FRAMES = [("left_wrist_yaw_link", np.array([0.10, 0.0, 0.0])),
               ("right_wrist_yaw_link", np.array([0.10, 0.0, 0.0]))]

# ---------------------------------------------------------------------------
# Command script
# ---------------------------------------------------------------------------
# Arm key poses (shoulder pitch/roll/yaw, elbow; left arm then right arm), the same joint
# configurations the SONIC / FALCON demos derive their EE targets from. An arm without a key
# configuration stays at AMO's default arm pose.
ARM_POSES = {
    "start": DEFAULT_DOF_POS[15:23],
    "both_hands_forward": np.array([-1.05, 0.15, 0.0, 0.45, -1.05, -0.15, 0.0, 0.45]),
    "right_hand_up": np.concatenate([DEFAULT_DOF_POS[15:19], [-2.5, -0.25, 0.0, 0.25]]),
    "arms_spread": np.array([-0.15, 1.25, 0.0, 0.25, -0.15, -1.25, 0.0, 0.25]),
}
# AMO command ranges (README): vx in [-0.5, 0.5], vy in [-0.4, 0.4] m/s
FWD_SPEED, SLOW_SPEED = 0.5, 0.4
# play_amo.py selects the in-place stand gait (and ignores the heading) when |vx| < 0.1, so
# side steps get the smallest forward command that still enables walking.
GAIT_MIN_VX = 0.1
EE_SEG = 2.4  # [s] per arm key pose; AMO blends arm targets in over 2 s

TORSO_KEYS = ("height", "torso_yaw", "torso_pitch", "torso_roll")
TORSO_RAMP = 0.8  # [s] smoothstep ramp of the torso commands at a segment change


def walk(speed, direction, turn=0.0):
    vx, vy = speed * math.cos(direction), speed * math.sin(direction)
    if abs(vx) < GAIT_MIN_VX:
        vx = GAIT_MIN_VX
    return dict(vx=vx, vy=vy, yaw_rate=turn)


def build_schedule(seed=0, torso=False):
    """List of (t_start, t_end, kind, cmd, label).

    cmd keys: vx, vy [m/s]; yaw_rate [rad/s]; arms (ARM_POSES key); height (offset from
    0.75 m), torso_yaw, torso_pitch, torso_roll [rad]. Missing keys mean zero; arms persist.
    """
    rng = np.random.default_rng(seed)
    sched = [
        (0.0, 1.0, "loco", dict(), "Stand"),
        (1.0, 3.2, "loco", walk(FWD_SPEED, 0.0), f"Walk forward (vx {FWD_SPEED} m/s)"),
        (3.2, 5.2, "loco", walk(SLOW_SPEED, math.pi / 2), f"Side-step left (vy {SLOW_SPEED} m/s)"),
        (5.2, 7.6, "loco", walk(SLOW_SPEED, math.pi), f"Walk backward (vx -{SLOW_SPEED} m/s)"),
        (7.6, 9.6, "loco", walk(SLOW_SPEED, -math.pi / 2), f"Side-step right (vy -{SLOW_SPEED} m/s)"),
    ]
    # random walk: random 45-deg-binned direction + random turning rate per segment
    t, t_end = 9.6, 12.9
    seg_len = (t_end - t) / 3
    dirs = np.arange(8) * math.pi / 4
    dir_names = ["fwd", "fwd-left", "left", "back-left", "back", "back-right", "right", "fwd-right"]
    for _ in range(3):
        k = int(rng.integers(0, 8))
        turn = float(rng.uniform(-0.9, 0.9))
        speed = FWD_SPEED if k == 0 else SLOW_SPEED
        sched.append((t, t + seg_len, "loco", walk(speed, float(dirs[k]), turn),
                      f"Random walk: {dir_names[k]} {speed} m/s, turn {turn:+.2f} rad/s"))
        t += seg_len
    sched.append((12.9, 13.6, "loco", dict(), "Stand"))
    # arm key poses while standing
    t = 13.6
    for pose, label in [("both_hands_forward", "Arms: both hands forward"), ("right_hand_up", "Arms: right hand up"),
                        ("arms_spread", "Arms: arms spread"), ("start", "Arms: back to start pose")]:
        sched.append((t, t + EE_SEG, "ee", dict(arms=pose), label))
        t += EE_SEG
    if torso:
        # AMO-specific torso commands (inputs of the adapter / policy)
        for dt, cmd, label in [
            (2.0, dict(height=-0.30), "Torso: squat to height 0.45 m"),
            (2.0, dict(torso_pitch=1.0), "Torso: stand up + bend forward (pitch 1.0 rad)"),
            (1.6, dict(torso_yaw=1.0), "Torso: yaw +1.0 rad"),
            (1.6, dict(torso_yaw=-1.0), "Torso: yaw -1.0 rad"),
            (1.6, dict(torso_roll=0.5), "Torso: roll +0.5 rad"),
            (2.8, dict(height=-0.25, torso_pitch=0.7, arms="both_hands_forward"),
             "Whole body: squat + bend forward + both hands forward"),
            (2.4, dict(arms="start"), "Back to start pose"),
        ]:
            sched.append((t, t + dt, "torso", cmd, label))
            t += dt
    return sched


def smoothstep(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


class CommandScript:
    """Writes play_amo.py `commands` and arm targets from the schedule, once per control tick."""

    def __init__(self, ctrl, schedule):
        self.ctrl, self.schedule = ctrl, schedule
        self.seg = None
        self.torso = {k: 0.0 for k in TORSO_KEYS}
        self.torso_from = dict(self.torso)
        self.arms = "start"
        self.target_yaw = 0.0

    def update(self, t):
        seg = next(s for s in self.schedule if s[0] <= t + 1e-9 < s[1] or s is self.schedule[-1])
        cmd = seg[3]
        if seg is not self.seg:
            self.seg = seg
            self.torso_from = dict(self.torso)
            if cmd.get("arms", self.arms) != self.arms:
                self.arms = cmd["arms"]
                self.ctrl.set_arm_target(ARM_POSES[self.arms])
        a = smoothstep((t - seg[0]) / TORSO_RAMP)
        for k in TORSO_KEYS:
            self.torso[k] = (1 - a) * self.torso_from[k] + a * cmd.get(k, 0.0)
        self.target_yaw += cmd.get("yaw_rate", 0.0) * CONTROL_DT
        c = self.ctrl.commands
        c[0] = cmd.get("vx", 0.0)
        c[1] = self.target_yaw
        c[2] = cmd.get("vy", 0.0)
        c[3] = self.torso["height"]
        c[4] = self.torso["torso_yaw"]
        c[5] = self.torso["torso_pitch"]
        c[6] = self.torso["torso_roll"]
        c[7] = 0.0  # no random arm actions (T toggle)
        return seg


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------
def add_sphere(scene, pos, rgba, radius=0.035):
    if scene.ngeom >= scene.maxgeom:
        return
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([radius, 0, 0]), np.asarray(pos, dtype=np.float64),
                        np.eye(3).reshape(-1), np.asarray(rgba, dtype=np.float32))
    scene.ngeom += 1


def add_arrow(scene, start, end, rgba, width=0.02):
    if scene.ngeom >= scene.maxgeom:
        return
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3), np.eye(3).reshape(-1),
                        np.asarray(rgba, dtype=np.float32))
    mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_ARROW, width, np.asarray(start, dtype=np.float64),
                         np.asarray(end, dtype=np.float64))
    scene.ngeom += 1


def draw_text(img, lines, org=(18, 36), scale=0.8):
    y = org[1]
    for i, (txt, color) in enumerate(lines):
        s = scale if i == 0 else scale * 0.8
        cv2.putText(img, txt, (org[0], y), cv2.FONT_HERSHEY_SIMPLEX, s, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, txt, (org[0], y), cv2.FONT_HERSHEY_SIMPLEX, s, color, 2, cv2.LINE_AA)
        y += int(36 * s + 6)


def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


class HandFK:
    """World positions of the hand key points, actual or for given arm joint angles."""

    def __init__(self, model):
        self.m, self.d = model, mujoco.MjData(model)
        self.ids = [model.body(n).id for n, _ in HAND_FRAMES]

    def points(self, d):
        return np.array([d.xpos[b] + d.xmat[b].reshape(3, 3) @ off for b, (_, off) in zip(self.ids, HAND_FRAMES)])

    def __call__(self, qpos, arm_q):
        self.d.qpos[:] = qpos
        self.d.qpos[-8:] = arm_q
        mujoco.mj_kinematics(self.m, self.d)
        return self.points(self.d)


def torso_in_heading_frame(d, pelvis_quat, torso_id):
    """Torso (yaw, pitch, roll) relative to the pelvis heading frame."""
    R = rot_z(-quatToEuler(pelvis_quat)[2]) @ d.xmat[torso_id].reshape(3, 3)
    return [math.atan2(R[1, 0], R[0, 0]), math.asin(-np.clip(R[2, 0], -1, 1)), math.atan2(R[2, 1], R[2, 2])]


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None, help="defaults to amo_mujoco_test/amo_mujoco_demo[_torso].mp4")
    ap.add_argument("--torso", action="store_true", help="append the AMO-specific torso command section")
    ap.add_argument("--duration", type=float, default=None, help="defaults to the end of the schedule")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda", help="amo_jit.pt was traced with tensors on cuda:0")
    ap.add_argument("--log", default=None, help="optional per-tick JSON log path")
    args = ap.parse_args()
    out = args.out or str(Path(__file__).resolve().parent /
                          ("amo_mujoco_demo_torso.mp4" if args.torso else "amo_mujoco_demo.mp4"))

    model, data = make_model(skybox=True)
    ctrl = AMOController(model, data, device=args.device, seed=args.seed)
    schedule = build_schedule(args.seed, torso=args.torso)
    script = CommandScript(ctrl, schedule)
    duration = args.duration if args.duration is not None else schedule[-1][1]
    for s in schedule:
        print(f"  [{s[0]:5.2f}, {s[1]:5.2f})  {s[4]}")

    pelvis = model.body("pelvis").id
    torso = model.body("torso_link").id
    hand_fk = HandFK(model)

    renderer = writer = None
    if not args.no_video:
        model.vis.global_.offwidth = max(model.vis.global_.offwidth, args.width)
        model.vis.global_.offheight = max(model.vis.global_.offheight, args.height)
        renderer = mujoco.Renderer(model, height=args.height, width=args.width)
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = True
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.distance, cam.azimuth, cam.elevation = 3.4, 150.0, -14.0
        cam.lookat[:] = [0, 0, 0.72]
        writer = imageio.get_writer(out, fps=args.fps, codec="libx264", quality=8,
                                    macro_block_size=8, ffmpeg_log_level="error",
                                    ffmpeg_params=["-threads", "4"])

    log = []
    n_ticks = int(round(duration / CONTROL_DT))
    next_frame_t = 0.0
    wall0 = time.time()
    fell = False

    for k in range(n_ticks):
        t = k * CONTROL_DT
        seg = script.update(t)
        kind, cmd, label = seg[2], seg[3], seg[4]
        c = ctrl.commands

        # ----------------------------------------------- policy (50 Hz) + physics (500 Hz) + video
        for _ in range(SIM_DECIMATION):
            ctrl.step()
            t_sim = ctrl.i * SIM_DT
            if writer is not None and t_sim >= next_frame_t - 1e-9:
                next_frame_t += 1.0 / args.fps
                pel = data.xpos[pelvis]
                cam.lookat[:] = 0.9 * cam.lookat + 0.1 * np.array([pel[0], pel[1], 0.72])
                renderer.update_scene(data, camera=cam)
                sc = renderer.scene
                heading = quatToEuler(data.qpos[3:7])[2]
                walking = abs(c[0]) >= GAIT_MIN_VX
                if walking:
                    # commanded movement direction (robot heading frame -> world)
                    mv = rot_z(heading) @ np.array([c[0], c[2], 0.0])
                    mv /= np.linalg.norm(mv)
                    st = np.array([pel[0], pel[1], 0.03]) + 0.25 * mv
                    add_arrow(sc, st, st + 0.7 * mv, [1.0, 0.55, 0.0, 0.95], 0.04)
                    # target heading commands[1] (tracked only while walking)
                    st = np.array([pel[0], pel[1], 0.02])
                    add_arrow(sc, st, st + 0.55 * np.array([math.cos(c[1]), math.sin(c[1]), 0.0]),
                              [0.2, 0.85, 1.0, 0.9], 0.025)
                show_hands = kind == "ee" or np.abs(ctrl.pd_target[15:] - ARM_POSES["start"]).max() > 0.05
                if kind == "torso":
                    # commanded torso axes (green, yaw/pitch/roll in the pelvis heading frame) vs actual (white)
                    o = data.xpos[torso]
                    R_cmd = rot_z(heading + float(c[4])) @ rot_y(float(c[5])) @ rot_x(float(c[6]))
                    R_act = data.xmat[torso].reshape(3, 3)
                    # connector lengths; MuJoCo draws arrows at about half of it, so the up axis clears the head
                    for axis, length in ((0, 1.0), (2, 1.7)):
                        add_arrow(sc, o, o + length * R_cmd[:, axis], [0.2, 0.95, 0.35, 0.6], 0.03)
                        add_arrow(sc, o, o + length * R_act[:, axis], [1, 1, 1, 0.9], 0.012)
                if show_hands:
                    # arm joint targets actually sent to the PD loop -> hand points (green L / red R), actual = white
                    tgt = hand_fk(data.qpos, ctrl.pd_target[15:])
                    act = hand_fk.points(data)
                    for i, col in ((0, [0.1, 0.9, 0.2, 0.8]), (1, [0.95, 0.2, 0.2, 0.8])):
                        add_sphere(sc, tgt[i], col, 0.045)
                        add_sphere(sc, act[i], [1, 1, 1, 0.9], 0.02)
                frame = renderer.render().copy()
                lines = [(f"AMO sim2sim (MuJoCo)   t = {t_sim:5.2f} s", (255, 255, 255)),
                         (f"Command: {label}", (255, 210, 60))]
                if kind == "ee":
                    lines.append(("Arms: joint targets (2 s blend) -> PD   "
                                  "green/red = L/R hand target, white = actual", (180, 230, 255)))
                elif kind == "torso":
                    lines.append((f"Torso: height {BASE_HEIGHT + c[3]:.2f} m, yaw {c[4]:+.2f}, pitch {c[5]:+.2f}, "
                                  f"roll {c[6]:+.2f} rad   green = commanded torso axes, white = actual",
                                  (180, 230, 255)))
                elif walking:
                    lines.append((f"Lower body: vx {c[0]:+.2f}, vy {c[2]:+.2f} m/s, target yaw {c[1]:+.2f} rad   "
                                  f"orange = commanded direction, cyan = target heading", (180, 230, 255)))
                else:
                    lines.append(("Lower body: stand (|vx| < 0.1 -> in-place gait phase)", (180, 230, 255)))
                draw_text(frame, lines)
                writer.append_data(frame)

        # ----------------------------------------------- logging
        pel = data.qpos[0:3].copy()
        hand_err = np.linalg.norm(hand_fk(data.qpos, ctrl.pd_target[15:]) - hand_fk.points(data), axis=1)
        hand_shift = np.linalg.norm(hand_fk(data.qpos, ctrl.arm_action) - hand_fk(data.qpos, ARM_POSES["start"]), axis=1)
        log.append(dict(t=round(t + CONTROL_DT, 3), seg=label, pelvis=pel.tolist(),
                        heading=float(quatToEuler(data.qpos[3:7])[2]),
                        torso_ypr=torso_in_heading_frame(data, data.qpos[3:7], torso),
                        cmd=c.tolist(), hand_err=hand_err.tolist(), hand_shift=hand_shift.tolist()))
        if pel[2] < 0.45 and not fell:
            fell = True
            print(f"!!! robot fell at t={t:.2f}s (pelvis z={pel[2]:.3f})")

    if writer is not None:
        writer.close()
        renderer.close()
        print(f"Wrote {out}")
    wall = time.time() - wall0
    print(f"Simulated {duration:.1f}s in {wall:.1f}s wall, fell={fell}")
    summarize(log, schedule)
    if args.log:
        with open(args.log, "w") as f:
            json.dump(log, f)


def summarize(log, schedule):
    print("\nPer-segment summary (displacement expressed in the robot's heading frame at segment start; "
          "torso yaw/pitch/roll relative to the pelvis heading at segment end):")
    print(f"{'segment':52s} {'fwd[m]':>7s} {'left[m]':>7s} {'dyaw[deg]':>9s} {'min z':>6s} "
          f"{'torso y/p/r[deg]':>17s} {'hand err L/R[cm]':>17s} {'target shift L/R[cm]':>21s}")
    for s in schedule:
        ents = [e for e in log if s[0] < e["t"] <= s[1] + 1e-9]
        if not ents:
            continue
        p0 = np.array(ents[0]["pelvis"])
        p1 = np.array(ents[-1]["pelvis"])
        h0 = ents[0]["heading"]
        dp = p1 - p0
        fwd = dp[0] * math.cos(h0) + dp[1] * math.sin(h0)
        left = -dp[0] * math.sin(h0) + dp[1] * math.cos(h0)
        dyaw = math.degrees((ents[-1]["heading"] - h0 + math.pi) % (2 * math.pi) - math.pi)
        minz = min(e["pelvis"][2] for e in ents)
        ypr = np.degrees(ents[-1]["torso_ypr"])
        err = 100 * np.mean([e["hand_err"] for e in ents[-10:]], axis=0)
        shift = 100 * np.array(ents[-1]["hand_shift"])
        print(f"{s[4]:52s} {fwd:7.2f} {left:7.2f} {dyaw:9.1f} {minz:6.3f} "
              f"{ypr[0]:+5.0f} {ypr[1]:+5.0f} {ypr[2]:+5.0f} {err[0]:9.1f} / {err[1]:4.1f} "
              f"{shift[0]:13.1f} / {shift[1]:4.1f}")


if __name__ == "__main__":
    main()
