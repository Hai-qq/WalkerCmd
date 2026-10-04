import numpy as np
from walkercmd.env import CmdWalker, Perturb


def test_reset_restores_model_params():
    e = CmdWalker(dr=True, seed=0)
    base = e.base_gear.copy()
    for _ in range(5):
        e.reset()
    # 随机化后参数应由 base 缩放得到，不会累积漂移
    ratio = e.model.actuator_gear[:, 0] / base[:, 0]
    assert np.allclose(ratio, ratio[0])
    e2 = CmdWalker(dr=False, seed=0); e2.reset()
    assert np.allclose(e2.model.actuator_gear, e2.base_gear)


def test_command_in_obs_and_tracking_reward_peaks_at_command():
    e = CmdWalker(v_cmd=1.2, seed=0)
    o, info = e.reset(seed=1)
    assert o.shape[0] == 18 and abs(o[-1] - 1.2) < 1e-9
    e.data.qvel[0] = 1.2
    _, r_on, *_ = e.step(np.zeros(6))
    e.reset(seed=1); e.data.qvel[0] = 0.0
    _, r_off, *_ = e.step(np.zeros(6))
    assert r_on > r_off


def test_push_applies_force_and_clears():
    e = CmdWalker(v_cmd=1.0, perturb=Perturb(push=100.0, push_every=10, push_len=3), seed=0)
    e.reset(seed=0)
    forces = []
    for _ in range(30):
        _, _, te, tr, _ = e.step(np.zeros(6))
        forces.append(float(abs(e.data.xfrc_applied[e.torso, 0])))
        if te or tr:
            break
    assert max(forces) == 100.0 and min(forces) == 0.0


def test_delay_shifts_actions():
    e = CmdWalker(v_cmd=1.0, perturb=Perturb(delay=2), seed=0)
    e.reset(seed=0)
    assert len(e._abuf) == 2
    e.step(np.ones(6)); assert len(e._abuf) == 2


def test_deterministic_given_seed():
    def run():
        e = CmdWalker(dr=True, seed=7); o, _ = e.reset(seed=7); out = []
        for _ in range(20):
            o, r, *_ = e.step(np.full(6, 0.3)); out.append(r)
        return out
    assert np.allclose(run(), run())
