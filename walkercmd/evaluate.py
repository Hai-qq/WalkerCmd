"""留出扰动评测：对每个模型逐因素扫描摩擦/质量/驱动增益/外推力/观测噪声/动作延迟。
训练域随机化范围外的水平（如摩擦 0.3、质量 0.6/1.4、推力 200/300N、延迟 3）用来测外推。
python -m walkercmd.evaluate --runs runs --out results/eval.csv
"""
from __future__ import annotations
import argparse, glob, os, itertools
from multiprocessing import Pool
import numpy as np
import pandas as pd
from walkercmd.env import CmdWalker, Perturb
from walkercmd.ppo import Policy

SWEEPS = {
    "nominal": [0],
    "friction": [0.3, 0.6, 1.0, 1.5],
    "mass": [0.6, 1.0, 1.4],
    "gear": [0.6, 0.8, 1.0, 1.2],
    "push": [100.0, 200.0, 300.0],
    "obs_noise": [0.02, 0.05],
    "delay": [1, 2, 3],
}
CMDS = (0.8, 1.2, 1.6)


def make_perturb(factor, level):
    if factor == "nominal":
        return Perturb()
    return Perturb(**{factor: level})


def run_job(args):
    path, factor, level, episodes = args
    pol = Policy(path)
    surv, rmse, tq, steps = [], [], [], []
    for ci, v in enumerate(CMDS):
        env = CmdWalker(perturb=make_perturb(factor, level), v_cmd=v, seed=123)
        for ep in range(episodes):
            o, _ = env.reset(seed=10_000 * ci + ep)
            errs, tqs, n, term = [], [], 0, False
            while True:
                o, r, te, tr, info = env.step(pol.act(o)); errs.append(info["err"]); tqs.append(info["torque2"]); n += 1
                if te or tr:
                    term = te; break
            surv.append(0.0 if term else 1.0); rmse.append(float(np.sqrt(np.mean(np.square(errs))))); tq.append(float(np.mean(tqs))); steps.append(n)
    cfg = os.path.basename(os.path.dirname(path))
    return {"model": cfg, "dr": cfg.startswith("dr"), "seed": int(cfg.split("_s")[1]), "factor": factor, "level": level,
            "survival": float(np.mean(surv)), "vel_rmse": float(np.mean(rmse)), "torque2": float(np.mean(tq)),
            "mean_steps": float(np.mean(steps)), "n_episodes": len(surv)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs"); ap.add_argument("--out", default="results/eval.csv")
    ap.add_argument("--episodes", type=int, default=8); ap.add_argument("--procs", type=int, default=16)
    a = ap.parse_args()
    models = sorted(glob.glob(os.path.join(a.runs, "*", "model.pt")))
    jobs = [(m, f, l, a.episodes) for m in models for f, ls in SWEEPS.items() for l in ls]
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with Pool(a.procs) as p:
        rows = p.map(run_job, jobs, chunksize=1)
    pd.DataFrame(rows).to_csv(a.out, index=False); print(len(rows), "rows ->", a.out)


if __name__ == "__main__":
    main()
