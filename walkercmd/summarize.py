"""汇总评测：按种子聚合（均值±标准差），输出 Markdown 表和鲁棒性曲线。"""
import sys
import pandas as pd, numpy as np
import itertools
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

df = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "results/eval.csv")
out = sys.argv[2] if len(sys.argv) > 2 else "results"
g = df.groupby(["factor", "level", "dr"]).agg(surv=("survival", "mean"), surv_sd=("survival", "std"),
                                              rmse=("vel_rmse", "mean"), rmse_sd=("vel_rmse", "std"), n_seeds=("seed", "nunique")).reset_index()
def perm_p(a, b):
    """精确置换检验（5 vs 5 个种子，双侧，检验均值差）。"""
    x = np.r_[a, b]; n = len(a); obs = abs(np.mean(a) - np.mean(b)); cnt = tot = 0
    for idx in itertools.combinations(range(len(x)), n):
        m = np.zeros(len(x), bool); m[list(idx)] = True
        cnt += abs(x[m].mean() - x[~m].mean()) >= obs - 1e-12; tot += 1
    return cnt / tot

lines = ["| 因素 | 水平 | 无DR 存活率 | DR 存活率 | 置换检验 p | 无DR 速度RMSE | DR 速度RMSE |", "|---|---|---|---|---|---|---|"]
order = ["nominal", "friction", "mass", "gear", "push", "obs_noise", "delay"]
for f in order:
    for l in sorted(g[g.factor == f].level.unique()):
        a = g[(g.factor == f) & (g.level == l) & (~g.dr)].iloc[0]; b = g[(g.factor == f) & (g.level == l) & (g.dr)].iloc[0]
        sa = df[(df.factor == f) & (df.level == l) & (~df.dr)].sort_values("seed").survival.values
        sb = df[(df.factor == f) & (df.level == l) & (df.dr)].sort_values("seed").survival.values
        p = perm_p(sa, sb)
        lines.append(f"| {f} | {l:g} | {a.surv:.2f}±{a.surv_sd:.2f} | {b.surv:.2f}±{b.surv_sd:.2f} | {p:.3f}{'*' if p <= 0.05 else ''} | {a.rmse:.2f}±{a.rmse_sd:.2f} | {b.rmse:.2f}±{b.rmse_sd:.2f} |")
open(f"{out}/summary.md", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
fs = [f for f in order if f != "nominal"]
fig, ax = plt.subplots(2, 3, figsize=(12, 6.5))
nom = g[g.factor == "nominal"]
for k, f in enumerate(fs):
    a = ax.flat[k]
    for dr, c, lab in [(False, "tab:gray", "no DR"), (True, "tab:blue", "DR")]:
        d = g[(g.factor == f) & (g.dr == dr)].sort_values("level")
        n0 = nom[nom.dr == dr].iloc[0]
        x = np.r_[0 if f in ("push", "obs_noise", "delay") else np.nan, d.level.values] if f in ("push", "obs_noise", "delay") else d.level.values
        y = np.r_[n0.surv, d.surv.values] if f in ("push", "obs_noise", "delay") else d.surv.values
        e = np.r_[n0.surv_sd, d.surv_sd.values] if f in ("push", "obs_noise", "delay") else d.surv_sd.values
        a.errorbar(x, y, yerr=e, marker="o", color=c, label=lab, capsize=3)
    a.set_title(f); a.set_ylim(-0.05, 1.05); a.set_ylabel("survival"); a.grid(alpha=.3)
ax.flat[0].legend()
plt.tight_layout(); plt.savefig(f"{out}/robustness.png", dpi=130)
