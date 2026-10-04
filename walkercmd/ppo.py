"""最小 PPO（连续动作，高斯策略）。观测与回报归一化、GAE、截断时自举。"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn


class RunningMeanStd:
    def __init__(self, shape=()):
        self.mean = np.zeros(shape, np.float64); self.var = np.ones(shape, np.float64); self.count = 1e-4

    def update(self, x):
        x = np.asarray(x, np.float64)
        bm, bv, bc = x.mean(0), x.var(0), x.shape[0]
        d = bm - self.mean; tot = self.count + bc
        self.mean = self.mean + d * bc / tot
        m2 = self.var * self.count + bv * bc + d ** 2 * self.count * bc / tot
        self.var = m2 / tot; self.count = tot

    def state(self):
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load(self, s):
        self.mean, self.var, self.count = s["mean"], s["var"], s["count"]


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=256):
        super().__init__()
        def mlp(out):
            return nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, out))
        self.pi, self.v = mlp(act_dim), mlp(1)
        self.log_std = nn.Parameter(torch.full((act_dim,), -0.5))
        for m in list(self.pi) + list(self.v):
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, np.sqrt(2)); nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.pi[-1].weight, 0.01); nn.init.orthogonal_(self.v[-1].weight, 1.0)

    def dist(self, o):
        return torch.distributions.Normal(self.pi(o), self.log_std.exp())

    def value(self, o):
        return self.v(o).squeeze(-1)


def normalize(obs, rms, clip=10.0):
    return np.clip((obs - rms.mean) / np.sqrt(rms.var + 1e-8), -clip, clip).astype(np.float32)


class Policy:
    """评测用：加载权重，确定性动作（取高斯均值）。"""
    def __init__(self, path):
        ck = torch.load(path, weights_only=False)
        self.net = ActorCritic(ck["obs_dim"], ck["act_dim"], ck.get("hidden", 256)); self.net.load_state_dict(ck["model"])
        self.rms = RunningMeanStd((ck["obs_dim"],)); self.rms.load(ck["rms"])

    @torch.no_grad()
    def act(self, obs):
        o = torch.as_tensor(normalize(obs, self.rms))
        return torch.tanh(self.net.pi(o)).numpy() if False else self.net.pi(o).clamp(-1, 1).numpy()
