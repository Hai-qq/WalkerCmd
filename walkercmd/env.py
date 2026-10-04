"""速度指令跟踪的双足环境（MuJoCo Walker2d）+ sim2real 风格的域随机化与扰动。

- 每个 episode 采样前进速度指令 v_cmd，观测 = 17 维状态 + v_cmd。
- 奖励：存活 + 速度跟踪（指数核）- 力矩代价。
- 域随机化（DR）：地面摩擦、连杆质量（含转动惯量）、关节驱动增益；
  另有观测噪声、动作延迟、外推力，用于评测时的"虚实差距"压力测试。
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import gymnasium as gym
from gymnasium import spaces


@dataclass
class Perturb:
    friction: float = 1.0      # 摩擦系数缩放
    mass: float = 1.0          # 质量/惯量缩放
    gear: float = 1.0          # 驱动增益缩放
    obs_noise: float = 0.0     # 观测高斯噪声标准差
    delay: int = 0             # 动作延迟（步）
    push: float = 0.0          # 外推力幅值（N，作用在躯干，水平方向）
    push_every: int = 150      # 每隔多少步推一次
    push_len: int = 5          # 每次持续步数


@dataclass
class DRRange:
    friction: tuple = (0.4, 1.6)
    mass: tuple = (0.7, 1.3)
    gear: tuple = (0.7, 1.3)
    obs_noise: tuple = (0.0, 0.02)
    delay: tuple = (0, 2)
    push: tuple = (0.0, 150.0)


V_RANGE = (0.5, 2.0)


class CmdWalker(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, dr: bool = False, dr_range: DRRange | None = None, perturb: Perturb | None = None,
                 v_cmd: float | None = None, seed: int | None = None, max_steps: int = 1000):
        self.env = gym.make("Walker2d-v5", max_episode_steps=max_steps + 1)
        self.u = self.env.unwrapped
        self.model, self.data = self.u.model, self.u.data
        self.dr, self.dr_range, self.fixed_perturb = dr, dr_range or DRRange(), perturb
        self.fixed_v = v_cmd
        self.max_steps = max_steps
        self.base_fric = self.model.geom_friction.copy()
        self.base_mass = self.model.body_mass.copy()
        self.base_inertia = self.model.body_inertia.copy()
        self.base_gear = self.model.actuator_gear.copy()
        self.torso = self.model.body("torso").id
        n = self.env.observation_space.shape[0]
        self.observation_space = spaces.Box(-np.inf, np.inf, (n + 1,), np.float64)
        self.action_space = self.env.action_space
        self.rng = np.random.default_rng(seed)
        self.p = Perturb()
        self.v_cmd = 1.0
        self.t = 0
        self._abuf: list[np.ndarray] = []

    # ---- 扰动采样与施加 ----
    def _sample_perturb(self) -> Perturb:
        if self.fixed_perturb is not None:
            return self.fixed_perturb
        if not self.dr:
            return Perturb()
        r, g = self.dr_range, self.rng
        return Perturb(friction=g.uniform(*r.friction), mass=g.uniform(*r.mass), gear=g.uniform(*r.gear),
                       obs_noise=g.uniform(*r.obs_noise), delay=int(g.integers(r.delay[0], r.delay[1] + 1)),
                       push=g.uniform(*r.push))

    def _apply(self, p: Perturb):
        self.model.geom_friction[:] = self.base_fric * p.friction
        self.model.body_mass[:] = self.base_mass * p.mass
        self.model.body_inertia[:] = self.base_inertia * p.mass
        self.model.actuator_gear[:] = self.base_gear * p.gear

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.p = self._sample_perturb()
        self._apply(self.p)
        obs, _ = self.env.reset(seed=int(self.rng.integers(1 << 31)))
        self.v_cmd = float(self.fixed_v if self.fixed_v is not None else self.rng.uniform(*V_RANGE))
        self.t = 0
        self._abuf = [np.zeros(self.action_space.shape[0], np.float32) for _ in range(self.p.delay)]
        self.data.xfrc_applied[:] = 0
        return self._obs(obs), {"v_cmd": self.v_cmd}

    def _obs(self, obs):
        o = np.asarray(obs, np.float64)
        if self.p.obs_noise > 0:
            o = o + self.rng.normal(0, self.p.obs_noise, o.shape)
        return np.concatenate([o, [self.v_cmd]])

    def step(self, action):
        a = np.clip(np.asarray(action, np.float32), -1, 1)
        if self.p.delay > 0:
            self._abuf.append(a)
            a_exec = self._abuf.pop(0)
        else:
            a_exec = a
        # 外推力：每 push_every 步，持续 push_len 步，方向随机（前/后）
        self.data.xfrc_applied[:] = 0
        if self.p.push > 0 and (self.t % self.p.push_every) < self.p.push_len and self.t > 0:
            if (self.t % self.p.push_every) == 0 or not hasattr(self, "_pdir"):
                self._pdir = 1.0 if self.rng.random() < 0.5 else -1.0
            self.data.xfrc_applied[self.torso, 0] = self._pdir * self.p.push
        obs, _, term, trunc, info = self.env.step(a_exec)
        vx = float(self.data.qvel[0])
        err = vx - self.v_cmd
        r = 1.0 + 2.0 * np.exp(-(err ** 2) / 0.25) - 1e-3 * float(np.sum(a_exec ** 2))
        self.t += 1
        trunc = trunc or self.t >= self.max_steps
        return self._obs(obs), r, bool(term), bool(trunc), {"vx": vx, "v_cmd": self.v_cmd, "err": err,
                                                              "torque2": float(np.sum(a_exec ** 2))}
