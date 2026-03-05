"""
test_gradient_flow.py — Algorithm correctness unit tests
=========================================================
Validates that:
1. MAPPO fc1/fc2 weights change during training (gradient flow)
2. IQL Q-network weights change during training
3. phi1_ablation uses obs[0] (R), not obs[2] (crisis_flag)
4. Environment _get_obs() returns [R, prev_mean_lambda, crisis_flag, t/T]
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from cleanrl_mappo_pgg import MLPActor, ppo_update_actor, relu, NNLayer
from envs.nonlinear_pgg_env import NonlinearPGGEnv

def test_mappo_gradient_flow():
    """fc1 and fc2 weights MUST change after a PPO update step."""
    rng = np.random.RandomState(42)
    actor = MLPActor(rng)
    
    # Snapshot weights before
    fc1_before = actor.fc1.W.copy()
    fc2_before = actor.fc2.W.copy()
    mh_before = actor.mean_head.W.copy()
    
    # Fake data: one step with nonzero advantage
    obs = np.array([0.5, 0.5, 0.0, 0.1], dtype=np.float32)
    act = 0.7
    old_lp = actor.log_prob(obs, act)
    adv = 1.0  # positive advantage
    
    ppo_update_actor(actor, [obs], [act], [old_lp], [adv])
    
    # All layers must have changed
    fc1_changed = not np.allclose(actor.fc1.W, fc1_before)
    fc2_changed = not np.allclose(actor.fc2.W, fc2_before)
    mh_changed = not np.allclose(actor.mean_head.W, mh_before)
    
    assert fc1_changed, "FAIL: fc1 weights did not change — gradient not flowing!"
    assert fc2_changed, "FAIL: fc2 weights did not change — gradient not flowing!"
    assert mh_changed, "FAIL: mean_head weights did not change!"
    print("  [PASS] MAPPO gradient flow: fc1, fc2, mean_head all updated")

def test_iql_weight_change():
    """IQL Q-network weights must change after training."""
    from cleanrl_qmix_pgg import QNet, relu
    rng = np.random.RandomState(42)
    q = QNet(rng)
    
    w_before = q.W1.copy()
    
    # Fake backward pass
    obs = np.array([0.5, 0.5, 0.0, 0.1], dtype=np.float32)
    q_vals = q.forward(obs)
    dq = np.zeros(11, dtype=np.float32)
    dq[5] = 1.0  # fake gradient on action 5
    grads = q.backward(dq)
    q.update(grads)
    
    assert not np.allclose(q.W1, w_before), "FAIL: IQL W1 did not change!"
    print("  [PASS] IQL Q-network weight update verified")

def test_obs_index_semantics():
    """obs[0] = R (resource), obs[2] = crisis_flag."""
    env = NonlinearPGGEnv()
    obs, _ = env.reset(seed=0)
    
    assert len(obs) == 4, f"FAIL: obs length {len(obs)}, expected 4"
    assert obs[0] == env.R, f"FAIL: obs[0]={obs[0]} != env.R={env.R}"
    assert obs[2] == float(env.R < env.r_crit), f"FAIL: obs[2] != crisis_flag"
    assert obs[3] == 0.0, f"FAIL: obs[3] != 0.0 at t=0"
    
    # R should be in [0, 1] range, crisis_flag is 0 or 1
    assert 0 <= obs[0] <= 1.0, f"FAIL: R={obs[0]} out of range"
    assert obs[2] in [0.0, 1.0], f"FAIL: crisis_flag={obs[2]} not boolean"
    print("  [PASS] obs index semantics: [R, prev_λ, crisis_flag, t/T]")

def test_phi1_uses_correct_obs():
    """phi1_ablation must use obs[0] (R), not obs[2] (crisis_flag)."""
    import phi1_ablation
    import inspect
    source = inspect.getsource(phi1_ablation.run_phi1)
    
    # Check that obs[0] is used for R_t, NOT obs[2]
    assert "obs[0]" in source, "FAIL: phi1_ablation doesn't use obs[0] for R_t!"
    assert "obs[2]" not in source or "crisis" in source.lower(), \
        "FAIL: phi1_ablation still uses obs[2] for R_t (should be obs[0])"
    print("  [PASS] phi1_ablation uses obs[0] (R) correctly")


if __name__ == "__main__":
    print("=" * 60)
    print("  Algorithm Correctness Unit Tests")
    print("=" * 60)
    
    tests = [
        test_mappo_gradient_flow,
        test_iql_weight_change,
        test_obs_index_semantics,
        test_phi1_uses_correct_obs,
    ]
    
    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except AssertionError as e:
            print(f"  [FAIL] {t.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"  [ERROR] {t.__name__}: {e}")
            failed += 1
    
    print(f"\n  Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    if failed > 0:
        sys.exit(1)
