"""Run AMO for a short standing test from a given whole-body pose and log the states (no video).

Physics and controller are those of amo_policy.py (play_amo.py port): AMO's g1.xml, 500 Hz
physics, 50 Hz policy, same PD gains and torque limits. Differences from the native harness:
  * Initial state: instead of the `home` keyframe + one mj_step, the robot starts at rest from the
    given pose (base at the origin, identity orientation, feet 1 mm above the floor).
  * Arm targets: arm_action = prev_arm_action = the pose's arm joints from t=0 (AMO's arm blend
    is then a no-op). DEFAULT_DOF_POS (observation / action offset) is unchanged.
Commands for the whole run: vx = vy = 0, target heading = initial heading, height offset 0
(0.75 m), torso yaw/pitch/roll 0, random-arm toggle off.

Usage (from the repo root):
    source amo_mujoco_test/cache_env.sh && source .venv/bin/activate
    python amo_mujoco_test/run_amo_stand_pose.py --pose <stand_pose.json> --log <out.npz>
"""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from amo_policy import (
    CONTROL_DT,
    DEFAULT_DOF_POS,
    DOF_NAMES,
    REPO,
    SIM_DECIMATION,
    SIM_DT,
    AMOController,
    quatToEuler,
)

XML = REPO / "g1.xml"
ARM_NAMES = DOF_NAMES[15:]


def load_pose(path):
    body = json.loads(Path(path).read_text())["body"]
    return np.array([body[f"{n}_joint"] for n in DOF_NAMES], dtype=np.float64)


def foot_collision_lowest_z(model, data):
    """Lowest point of the collision geoms attached to the ankle_roll links."""
    z = np.inf
    for g in range(model.ngeom):
        body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[g]) or ""
        if not body.endswith("ankle_roll_link") or not (model.geom_contype[g] or model.geom_conaffinity[g]):
            continue
        assert model.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE, "foot collision geoms are spheres in g1.xml"
        z = min(z, data.geom_xpos[g][2] - model.geom_size[g][0])
    return z


def reset_to_pose(model, data, q23):
    mujoco.mj_resetData(model, data)
    data.qpos[0:3] = 0.0
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    data.qpos[7:] = q23
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    data.qpos[2] -= foot_collision_lowest_z(model, data) - 0.001
    mujoco.mj_forward(model, data)


def robot_self_contacts(model, data):
    out = []
    for c in data.contact[: data.ncon]:
        b1, b2 = model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2]
        if b1 == 0 or b2 == 0:
            continue
        out.append([mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b1),
                    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b2), float(c.dist)])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pose", required=True, help="JSON with a 'body' dict of joint name -> angle [rad]")
    ap.add_argument("--log", required=True, help="output .npz")
    ap.add_argument("--duration", type=float, default=2.0)
    ap.add_argument("--device", default="cuda", help="amo_jit.pt was traced with tensors on cuda:0")
    args = ap.parse_args()

    q23 = load_pose(args.pose)
    model = mujoco.MjModel.from_xml_path(str(XML))
    model.opt.timestep = SIM_DT
    data = mujoco.MjData(model)
    reset_to_pose(model, data, q23)

    ctrl = AMOController(model, data, device=args.device)
    ctrl.arm_action = q23[15:].copy()
    ctrl.prev_arm_action = q23[15:].copy()
    heading0 = float(quatToEuler(data.qpos[3:7])[2])
    ctrl.commands[:] = 0.0
    ctrl.commands[1] = heading0

    n_ticks = int(round(args.duration / CONTROL_DT))
    ts, qposes, targets, contacts = [], [], [], []
    for k in range(n_ticks + 1):
        ts.append(k * CONTROL_DT)
        qposes.append(data.qpos.copy())
        contacts.append(robot_self_contacts(model, data))
        if k == n_ticks:
            # command the controller would send at the final state (logged only, not simulated)
            ctrl.extract_data()
            ctrl.policy_tick()
            targets.append(ctrl.pd_target[15:].copy())
            break
        for s in range(SIM_DECIMATION):
            ctrl.step()
            if s == 0:
                targets.append(ctrl.pd_target[15:].copy())

    qpos = np.array(qposes)
    meta = dict(
        controller="C8 AMO",
        sim_dt=SIM_DT, control_dt=CONTROL_DT, sim_decimation=SIM_DECIMATION,
        commands=dict(vx=0.0, vy=0.0, target_heading=heading0, height_offset=0.0, height=0.75,
                      torso_yaw=0.0, torso_pitch=0.0, torso_roll=0.0, random_arm_toggle=0),
        arm_targets=dict(zip(ARM_NAMES, q23[15:].tolist())),
        default_dof_pos_unchanged=DEFAULT_DOF_POS.tolist(),
        deviations=[
            "Initial state: pose from --pose at rest (base at origin, identity quat, foot collision spheres "
            "1 mm above the floor) instead of the 'home' keyframe + one mj_step.",
            "Arm targets: arm_action = prev_arm_action = pose arms from t=0 (native startup blends from "
            "DEFAULT_DOF_POS[15:]; here the blend is a no-op).",
            "Robot has no wrist joints and no Dex3 hands (AMO's own 23-DoF g1.xml with rubber hands); "
            "the pose's wrist and finger values are not applied.",
        ],
        log_rows="row k = state at t = k*control_dt before the k-th policy tick; arm_pd_target row k = "
                 "arm PD target computed at that tick (last row: computed at the final state, not simulated)",
    )
    out = Path(args.log)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, t=np.array(ts), qpos=qpos, xml=str(XML), arm_joint_names=np.array([f"{n}_joint" for n in ARM_NAMES]),
             arm_pd_target=np.array(targets), self_contacts=json.dumps(contacts), meta=json.dumps(meta))

    # summary
    base_xy = qpos[:, 0:2] - qpos[0, 0:2]
    rpy = quatToEuler(qpos[-1, 3:7])
    tilt = np.degrees(np.arccos(np.clip(np.cos(rpy[0]) * np.cos(rpy[1]), -1, 1)))
    dq = np.abs(qpos[-1, 7:] - q23)
    n_contact_steps = sum(1 for c in contacts if c)
    print(f"saved {out}  (N={len(ts)}, nq={model.nq})")
    print(f"pelvis z: t=0 {qpos[0, 2]:.4f}  t={ts[-1]:.2f} {qpos[-1, 2]:.4f}  min {qpos[:, 2].min():.4f}")
    print(f"base xy drift: final {np.linalg.norm(base_xy[-1]) * 100:.2f} cm, max {np.linalg.norm(base_xy, axis=1).max() * 100:.2f} cm")
    print(f"pelvis tilt at end: {tilt:.2f} deg (roll {np.degrees(rpy[0]):.2f}, pitch {np.degrees(rpy[1]):.2f})")
    print(f"max |q - pose| at end: arms {dq[15:].max():.4f} rad ({ARM_NAMES[int(dq[15:].argmax())]}), "
          f"legs+waist {dq[:15].max():.4f} rad ({DOF_NAMES[int(dq[:15].argmax())]})")
    print(f"robot self-contacts: {n_contact_steps} of {len(contacts)} logged steps")
    print("final legs+waist:", np.round(qpos[-1, 7:22], 3).tolist())
    print("final arms:", np.round(qpos[-1, 22:], 3).tolist())


if __name__ == "__main__":
    main()
