"""Compliant root drive for the free-floating LEAP hand (live_sim_view.py --floating).

Writing the tracked wrist pose straight into the simulation (``write_root_pose_to_sim``) makes the
palm *kinematic*: it goes wherever the human's hand is, through the nut, the screw and the table,
because nothing in PhysX may push back on a teleport. Only the fingers, which hang off PD joints,
can yield. For grasping/screwing the whole hand must be able to stop at an object, so in
``--dynamic-objects`` mode the root is a normal dynamic body pulled toward the tracked pose by a
6-DoF spring-damper (a wrench on the root link), with the force capped at what a human arm would
push with. Blocked by an object, the hand lags behind the tracker instead of sinking into it.

Gains are set from the hand's own mass so the response is a critically damped ``ROOT_DRIVE_HZ``
second-order system: stiff enough to track a 10 Hz teleop stream without visible lag, soft enough
for a 1/120 s explicit-force update.
"""

from __future__ import annotations

import math

import torch

ROOT_DRIVE_HZ = 6.0            # natural frequency of the pose tracking (Hz)
ROOT_DRIVE_ZETA = 1.0          # damping ratio (1 = critically damped, no overshoot)
ROOT_DRIVE_MAX_FORCE = 60.0    # N   - a human arm pressing, not a robot ramming
ROOT_DRIVE_MAX_TORQUE = 4.0    # N m
ROOT_DRIVE_RADIUS = 0.08       # m   - hand mass lumped at this radius for the rotational inertia guess


class RootPoseDrive:
    """Pull an articulation's root link toward a target pose with a capped spring-damper wrench."""

    def __init__(self, robot, hz: float = ROOT_DRIVE_HZ, zeta: float = ROOT_DRIVE_ZETA,
                 max_force: float = ROOT_DRIVE_MAX_FORCE, max_torque: float = ROOT_DRIVE_MAX_TORQUE):
        self.robot = robot
        self.device = robot.device
        mass = float(robot.data.default_mass[0].sum())              # whole hand
        inertia = mass * ROOT_DRIVE_RADIUS ** 2
        wn = 2.0 * math.pi * hz
        self.kp = mass * wn * wn
        self.kd = 2.0 * zeta * mass * wn
        self.kr = inertia * wn * wn
        self.kdr = 2.0 * zeta * inertia * wn
        self.max_force = max_force
        self.max_torque = max_torque
        self.mass = mass
        # the root link is body 0 of the articulation
        self._body_ids = [0]
        self._forces = torch.zeros((1, 1, 3), device=self.device)
        self._torques = torch.zeros((1, 1, 3), device=self.device)
        self.target: torch.Tensor | None = None                      # [1, 7] pos + quat wxyz

    def describe(self) -> str:
        return (f"root drive: mass {self.mass:.3f} kg, kp {self.kp:.0f} N/m, kd {self.kd:.1f} N s/m, "
                f"kr {self.kr:.2f} N m/rad, cap {self.max_force:.0f} N / {self.max_torque:.1f} N m")

    def set_target(self, pose_wxyz: torch.Tensor) -> None:
        self.target = pose_wxyz.to(self.device)

    @staticmethod
    def _quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        aw, ax, ay, az = a.unbind(-1)
        bw, bx, by, bz = b.unbind(-1)
        return torch.stack([
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ], dim=-1)

    def rotation_error(self) -> torch.Tensor:
        """World-frame rotation vector taking the current root orientation to the target."""
        q = self.robot.data.root_quat_w[0:1]                          # wxyz
        qt = self.target[0:1, 3:7]
        q_conj = q * torch.tensor([1.0, -1.0, -1.0, -1.0], device=self.device)
        dq = self._quat_mul(qt, q_conj)                               # q_err = q_target * q^-1 (world frame)
        dq = torch.where(dq[:, 0:1] < 0, -dq, dq)                     # shortest arc
        w = dq[:, 0].clamp(-1.0, 1.0)
        v = dq[:, 1:4]
        s = v.norm(dim=-1, keepdim=True)
        angle = 2.0 * torch.atan2(s, w.unsqueeze(-1))
        axis = torch.where(s > 1e-9, v / s.clamp_min(1e-9), torch.zeros_like(v))
        return axis * angle

    def apply(self) -> tuple[float, float]:
        """Compute and queue this step's wrench (call before ``robot.write_data_to_sim``).

        Returns the applied |force| and |torque| for diagnostics."""
        if self.target is None:
            return 0.0, 0.0
        pos = self.robot.data.root_pos_w[0:1]
        lin_vel = self.robot.data.root_lin_vel_w[0:1]
        ang_vel = self.robot.data.root_ang_vel_w[0:1]
        force = self.kp * (self.target[0:1, 0:3] - pos) - self.kd * lin_vel
        torque = self.kr * self.rotation_error() - self.kdr * ang_vel
        fn = force.norm()
        if fn > self.max_force:
            force = force * (self.max_force / fn)
        tn = torque.norm()
        if tn > self.max_torque:
            torque = torque * (self.max_torque / tn)
        self._forces[0, 0] = force[0]
        self._torques[0, 0] = torque[0]
        self.robot.set_external_force_and_torque(self._forces, self._torques, body_ids=self._body_ids)
        return float(force.norm()), float(torque.norm())

    def position_error(self) -> float:
        if self.target is None:
            return 0.0
        return float((self.target[0, 0:3] - self.robot.data.root_pos_w[0]).norm())
