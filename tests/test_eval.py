import os, tempfile, torch, numpy as np
from walkercmd.ppo import ActorCritic, RunningMeanStd
from walkercmd.evaluate import run_job


def _fake_model(tmp):
    net = ActorCritic(18, 6); rms = RunningMeanStd((18,))
    p = os.path.join(tmp, "dr_s0"); os.makedirs(p)
    torch.save({"model": net.state_dict(), "rms": rms.state(), "obs_dim": 18, "act_dim": 6, "hidden": 256}, os.path.join(p, "model.pt"))
    return os.path.join(p, "model.pt")


def test_eval_job_is_deterministic_and_bounded():
    with tempfile.TemporaryDirectory() as d:
        m = _fake_model(d)
        a = run_job((m, "push", 200.0, 1)); b = run_job((m, "push", 200.0, 1))
        assert a == b
        assert 0.0 <= a["survival"] <= 1.0 and a["vel_rmse"] >= 0
        assert a["n_episodes"] == 3
