"""PPO 训练：python -m walkercmd.train --dr 1 --seed 0 --steps 1500000 --out runs/dr_s0"""
from __future__ import annotations
import argparse, csv, os, time
import numpy as np
import torch
from walkercmd.env import CmdWalker
from walkercmd.ppo import ActorCritic, RunningMeanStd, normalize


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dr", type=int, default=0); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=1_500_000); ap.add_argument("--out", required=True)
    ap.add_argument("--n_envs", type=int, default=8); ap.add_argument("--n_steps", type=int, default=256)
    a = ap.parse_args()
    torch.set_num_threads(1); torch.manual_seed(a.seed); np.random.seed(a.seed)
    os.makedirs(a.out, exist_ok=True)
    envs = [CmdWalker(dr=bool(a.dr), seed=a.seed * 1000 + i) for i in range(a.n_envs)]
    obs = np.stack([e.reset(seed=a.seed * 1000 + i)[0] for i, e in enumerate(envs)])
    obs_dim, act_dim = obs.shape[1], envs[0].action_space.shape[0]
    net = ActorCritic(obs_dim, act_dim); opt = torch.optim.Adam(net.parameters(), lr=3e-4, eps=1e-5)
    rms, ret_rms = RunningMeanStd((obs_dim,)), RunningMeanStd(())
    gamma, lam, clip, epochs, mbs = 0.99, 0.95, 0.2, 10, 8
    B = a.n_envs * a.n_steps; n_upd = a.steps // B
    ret_acc = np.zeros(a.n_envs); ep_ret = np.zeros(a.n_envs); ep_len = np.zeros(a.n_envs)
    done_ret, done_len = [], []
    f = open(os.path.join(a.out, "log.csv"), "w", newline=""); w = csv.writer(f); w.writerow(["steps", "ep_return", "ep_len", "sec"])
    t0 = time.time(); steps = 0
    for u in range(n_upd):
        for g in opt.param_groups: g["lr"] = 3e-4 * (1 - u / n_upd)
        O = np.zeros((a.n_steps, a.n_envs, obs_dim), np.float32); A = np.zeros((a.n_steps, a.n_envs, act_dim), np.float32)
        LP = np.zeros((a.n_steps, a.n_envs), np.float32); R = np.zeros((a.n_steps, a.n_envs), np.float32)
        D = np.zeros((a.n_steps, a.n_envs), np.float32); V = np.zeros((a.n_steps + 1, a.n_envs), np.float32)
        for t in range(a.n_steps):
            rms.update(obs); no = normalize(obs, rms)
            with torch.no_grad():
                ot = torch.as_tensor(no); d = net.dist(ot); act = d.sample(); lp = d.log_prob(act).sum(-1); V[t] = net.value(ot).numpy()
            O[t], A[t], LP[t] = no, act.numpy(), lp.numpy()
            nxt = np.zeros_like(obs); rew = np.zeros(a.n_envs, np.float32); dn = np.zeros(a.n_envs, np.float32)
            for i, e in enumerate(envs):
                o2, r, te, tr, _ = e.step(act[i].numpy())
                ep_ret[i] += r; ep_len[i] += 1; ret_acc[i] = ret_acc[i] * gamma + r
                if te or tr:
                    if tr and not te:  # 时间截断：用价值自举
                        with torch.no_grad():
                            r += gamma * float(net.value(torch.as_tensor(normalize(o2[None], rms)))[0])
                    done_ret.append(ep_ret[i]); done_len.append(ep_len[i]); ep_ret[i] = 0; ep_len[i] = 0; ret_acc[i] = 0
                    o2, _ = e.reset(); dn[i] = 1.0
                nxt[i], rew[i] = o2, r
            ret_rms.update(ret_acc.copy()); R[t] = rew / np.sqrt(ret_rms.var + 1e-8); D[t] = dn; obs = nxt
        steps += B
        with torch.no_grad():
            V[-1] = net.value(torch.as_tensor(normalize(obs, rms))).numpy()
        adv = np.zeros_like(R); last = 0
        for t in reversed(range(a.n_steps)):
            delta = R[t] + gamma * V[t + 1] * (1 - D[t]) - V[t]
            last = delta + gamma * lam * (1 - D[t]) * last; adv[t] = last
        ret = adv + V[:-1]
        bo, ba, blp = (torch.as_tensor(x.reshape(B, -1)) for x in (O, A, LP.reshape(B, 1)))
        blp = blp.squeeze(-1); badv = torch.as_tensor(adv.reshape(B)); bret = torch.as_tensor(ret.reshape(B))
        for _ in range(epochs):
            perm = torch.randperm(B)
            for k in range(mbs):
                idx = perm[k * (B // mbs):(k + 1) * (B // mbs)]
                d = net.dist(bo[idx]); lp = d.log_prob(ba[idx]).sum(-1); ratio = (lp - blp[idx]).exp()
                ad = badv[idx]; ad = (ad - ad.mean()) / (ad.std() + 1e-8)
                pl = -torch.min(ratio * ad, ratio.clamp(1 - clip, 1 + clip) * ad).mean()
                vl = 0.5 * (net.value(bo[idx]) - bret[idx]).pow(2).mean()
                opt.zero_grad(); (pl + 0.5 * vl).backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5); opt.step()
        if u % 5 == 0 or u == n_upd - 1:
            er = float(np.mean(done_ret[-30:])) if done_ret else 0.0; el = float(np.mean(done_len[-30:])) if done_len else 0.0
            w.writerow([steps, round(er, 2), round(el, 1), round(time.time() - t0)]); f.flush()
    torch.save({"model": net.state_dict(), "rms": rms.state(), "obs_dim": obs_dim, "act_dim": act_dim, "hidden": 256,
                "dr": a.dr, "seed": a.seed, "steps": steps}, os.path.join(a.out, "model.pt"))
    f.close()


if __name__ == "__main__":
    main()
