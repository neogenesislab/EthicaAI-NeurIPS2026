"""
Genesis v2.0 End-to-End Pipeline Test
모든 v2.0 모듈이 GPU 환경에서 올바르게 작동하는지 검증.
"""
import sys
sys.path.insert(0, ".")

import jax
import jax.numpy as jnp
import time

print("=" * 60)
print("  Genesis v2.0 End-to-End Pipeline Test")
print("=" * 60)
print(f"  JAX: {jax.__version__}")
print(f"  Backend: {jax.default_backend()}")
print(f"  Devices: {jax.devices()}")
print()

results = []

# 1. Config + IA defaults
try:
    from simulation.jax.config import get_config
    config = get_config("small")
    assert "IA_ALPHA" in config
    assert "USE_INEQUITY_AVERSION" in config
    assert "MEDIATOR_K" in config
    print(f"[1/6] ✅ Config: IA_ALPHA={config['IA_ALPHA']}, IA_BETA={config['IA_BETA']}, MEDIATOR_K={config['MEDIATOR_K']}")
    results.append(True)
except Exception as e:
    print(f"[1/6] ❌ Config: {e}")
    results.append(False)

# 2. reward_shaping (IA + vmap)
try:
    from simulation.genesis.reward_shaping import transform_rewards
    B, N = 4, 5
    r = jnp.array([[10.0, 1.0, 1.0, 1.0, 1.0]] * B)
    s = r.copy()
    ia_config = {"USE_INEQUITY_AVERSION": True, "IA_ALPHA": 5.0, "IA_BETA": 0.05, "IA_EMA_LAMBDA": 0.95}
    
    def _ia_per_env(rn, sn):
        t, _ = transform_rewards(rn, sn, ia_config, N)
        return t
    
    ia_out = jax.vmap(_ia_per_env)(r, s)
    assert ia_out.shape == (B, N), f"Shape mismatch: {ia_out.shape}"
    agent0_penalty = float(r[0, 0] - ia_out[0, 0])
    print(f"[2/6] ✅ reward_shaping: shape={ia_out.shape}, agent0_guilt_penalty={agent0_penalty:.4f}")
    results.append(True)
except Exception as e:
    print(f"[2/6] ❌ reward_shaping: {e}")
    results.append(False)

# 3. sanctioning
try:
    from simulation.genesis.sanctioning import apply_sanction, redistribute_rewards, detect_defectors
    rewards = jnp.array([1.0, 0.5, 0.8, 0.3, 0.6])
    poked = apply_sanction(rewards, 0, 3, "poke")
    redist = redistribute_rewards(rewards)
    assert abs(float(jnp.sum(rewards)) - float(jnp.sum(redist))) < 0.01
    print(f"[3/6] ✅ sanctioning: poke={list(poked)}, conserved={abs(float(jnp.sum(rewards)) - float(jnp.sum(redist))) < 0.01}")
    results.append(True)
except Exception as e:
    print(f"[3/6] ❌ sanctioning: {e}")
    results.append(False)

# 4. mediator
try:
    from simulation.genesis.mediator import Mediator
    med = Mediator({"MEDIATOR_K": 5})
    for _ in range(12):
        med.should_consult()
    actions = med.compute_collective_action(10)
    report = med.get_report()
    assert actions.shape == (10,)
    print(f"[4/6] ✅ mediator: actions_shape={actions.shape}, step={report['step_counter']}")
    results.append(True)
except Exception as e:
    print(f"[4/6] ❌ mediator: {e}")
    results.append(False)

# 5. tree_search
try:
    from simulation.genesis.tree_search import AgenticTreeSearch
    ats = AgenticTreeSearch(tree_path="/tmp/test_tree_v2.json")
    root = ats.create_root({"GENESIS_BETA": 1.0}, hypothesis="Base Config")
    children = ats.expand(root, 2)
    ats.evaluate_node(children[0], 0.35)
    ats.evaluate_node(children[1], 0.15)
    stats = ats.get_stats()
    assert stats["total_nodes"] >= 3
    print(f"[5/6] ✅ tree_search: nodes={stats['total_nodes']}, best_cr={stats['best_cr']:.4f}")
    results.append(True)
except Exception as e:
    print(f"[5/6] ❌ tree_search: {e}")
    results.append(False)

# 6. train_pipeline import + IA flag check
try:
    from simulation.jax.training.train_pipeline import make_train, _IA_AVAILABLE
    assert _IA_AVAILABLE, "IA module not imported in train_pipeline!"
    config = get_config("small")
    config["USE_INEQUITY_AVERSION"] = True
    config["SEEDS"] = [42]
    train_fn = make_train(config)
    print(f"[6/6] ✅ train_pipeline: make_train OK, _IA_AVAILABLE={_IA_AVAILABLE}")
    results.append(True)
except Exception as e:
    print(f"[6/6] ❌ train_pipeline: {e}")
    results.append(False)

# Summary
print()
print("=" * 60)
passed = sum(results)
total = len(results)
if passed == total:
    print(f"  🎉 ALL {total} TESTS PASSED — Genesis v2.0 Pipeline Ready!")
else:
    print(f"  ⚠️ {passed}/{total} tests passed, {total-passed} failed")
print("=" * 60)
