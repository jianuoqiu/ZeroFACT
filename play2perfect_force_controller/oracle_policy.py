"""The released RL policy served as a chunk policy through the simulator (import after ``AppLauncher``).

Why: the middle layer expects, every ``chunk`` frames, ``horizon`` future robot states plus the
force targets that go with them. play2perfect's policy is a single-step, closed-loop controller
(it re-observes the object every 16 ms and outputs one action) and predicts no forces. This
wrapper turns it into exactly the chunk policy the middle layer is built for:

    at a chunk boundary   snapshot the world (physics, env bookkeeping, sensors, RNG, LSTM state),
                          roll the RL policy forward ``horizon`` frames inside the simulator,
                          recording per frame the command it applied, the joint state it reached,
                          the net fingertip forces and the pad contact centroids, then put the
                          world back;
    in between            the middle layer executes the chunk (with or without the force law);
                          the env's task logic keeps running (``after_frame``) and the policy's
                          recurrent state follows the EXECUTED trajectory (``advance``).

So the force targets are not predicted by a network: they are what the policy would have
produced over the next ``horizon`` frames had it kept running - the same quantity the replay
policy takes from the recording, re-planned from the current state instead of fixed at episode
start. It is an optimistic stand-in for a learned chunk policy (its chunks are consistent with
the true dynamics) and it needs the simulator to plan, so it is an evaluation device, not a
deployable policy. Its rollouts are also the (observation -> state chunk + force chunk) pairs a
learned chunk policy would be trained on.

Caveat: PhysX's contact cache is not part of the restorable state, so a restore is exact for
every buffer that can be read back, but the first solver step afterwards starts from cold
contacts. The floor of that perturbation is what the chunk-1 exact-command run measures
(``run_tracking.py --policy oracle --exact-replay --chunk 1 --horizon 1``): with a perfect restore
it would reproduce the recording bit for bit.
"""

from __future__ import annotations

import time

import numpy as np
import torch

from .config import PolicyConfig
from .episode import pad_contact_centroid
from .policy import BasePolicy, PolicyOutput


class OracleChunkPolicy(BasePolicy):
    """play2perfect's RL policy + the simulator as the middle layer's chunk policy."""

    def __init__(self, env, player, wrapped, cfg: PolicyConfig, deterministic: bool = True):
        super().__init__(cfg)
        self.env = env
        self.player = player            # rl_games player (LSTM: ``player.states`` is its memory)
        self.wrapped = wrapped          # the rl_games env wrapper (observation formatting/clipping)
        self.deterministic = deterministic
        self.num_plans = 0
        self.plan_time_s = 0.0
        self.planned_done_frames: list[int] = []      # frames at which a plan saw the env finish

    # ------------------------------------------------------------------ policy-side state
    def reset(self) -> None:
        self.player.reset()             # recurrent state -> zeros, as at the recording's frame 0
        self.num_plans = 0
        self.plan_time_s = 0.0
        self.planned_done_frames = []

    def _observation(self) -> torch.Tensor:
        """The policy input for the CURRENT state, formatted exactly as ``player.env_step`` does."""
        obs = self.player.obs_to_torch(self.wrapped._process_obs(self.env.observe()))
        embd = getattr(self.player, "intr_reward_coef_embd", None)     # SAPG block-id column
        return obs if embd is None else torch.cat([obs, embd], dim=1)

    def advance(self) -> None:
        """Executing-side recurrent update, called at the start of every executed frame: the
        policy sees the observation of the frame about to be executed (at a boundary the same one
        the planner saw first), so its hidden state follows the executed trajectory, not the plan."""
        if getattr(self.player, "is_rnn", False):
            self.player.get_action(self._observation(), is_deterministic=self.deterministic)

    def after_frame(self, joint_target: np.ndarray) -> dict:
        """Run the env's task logic for an executed frame (``AssemblyBench.post_frame_bookkeeping``)."""
        target = torch.as_tensor(np.asarray(joint_target, dtype=np.float32),
                                 device=self.env.device).unsqueeze(0)
        return self.env.post_frame_bookkeeping(target)

    # ------------------------------------------------------------------ planning
    def predict(self, frame: int, observation: dict | None = None) -> PolicyOutput:
        env, player = self.env, self.player
        t0 = time.perf_counter()
        horizon = int(self.cfg.horizon)
        n_joints = int(env.robot.num_joints)
        n_tips = len(env.contacts)
        joint_pos = np.zeros((horizon, n_joints))
        joint_target = np.zeros((horizon, n_joints))
        force = np.zeros((horizon, n_tips, 3))
        point = np.full((horizon, n_tips, 3), np.nan)
        done_at = None
        successes = None
        snap = env.snapshot_full()
        rnn = [s.clone() for s in player.states] if getattr(player, "is_rnn", False) else None
        env.planning = True
        try:
            obs = self._observation()
            for j in range(horizon):
                action = player.get_action(obs, is_deterministic=self.deterministic)
                obs, _rew, dones, _info = player.env_step(self.wrapped, action)
                joint_target[j] = env._cur_targets[0].cpu().numpy()         # command held during the frame
                joint_pos[j] = env.robot.data.joint_pos[0].cpu().numpy()     # state at the frame end
                f_net, pair_f, pair_p = env.tactile_latest()
                force[j] = f_net
                point[j] = pad_contact_centroid(pair_f, pair_p)
                if done_at is None and bool(torch.as_tensor(dones).reshape(-1)[0].item()):
                    done_at = j
            successes = int(env._successes[0].item())
        finally:
            env.planning = False
            env.restore_full(snap)
            if rnn is not None:
                player.states = rnn
        self.num_plans += 1
        self.plan_time_s += time.perf_counter() - t0
        if done_at is not None:
            self.planned_done_frames.append(int(frame) + done_at)
        states = joint_pos if self.cfg.state_source == "joint_pos" else joint_target
        return PolicyOutput(
            start_frame=int(frame),
            joint_pos=states.copy(),
            fingertip_force=np.linalg.norm(force, axis=-1),
            fingertip_force_vec=force,
            contact_point=point,
            info={"source": "oracle", "state_source": self.cfg.state_source,
                  "planned_done_at": done_at, "planned_successes": successes,
                  # the plan itself, for the lead diagnostic: what the RL policy commanded vs
                  # what it reached (its own preload), per planned frame
                  "plan_joint_target": joint_target.copy(), "plan_joint_pos": joint_pos.copy(),
                  "plan_force": force.copy()},
        )

    @property
    def stats(self) -> dict:
        return {
            "plans": self.num_plans,
            "plan_time_s": round(self.plan_time_s, 2),
            "mean_plan_ms": round(1000.0 * self.plan_time_s / max(1, self.num_plans), 1),
            "planned_done_frames": self.planned_done_frames[:20],
        }
