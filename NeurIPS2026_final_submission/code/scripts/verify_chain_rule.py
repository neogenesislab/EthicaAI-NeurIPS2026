"""
verify_chain_rule.py — MAPPO backward pass mathematical verification
=====================================================================
Numerically verifies the analytical chain rule gradient against
finite-difference gradient for fc1, fc2, and mean_head.
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from cleanrl_mappo_pgg import MLPActor, relu, NNLayer

def finite_diff_gradient(actor, obs, act, param_name, layer_attr, eps=1e-5):
    """Compute numerical gradient of log_prob w.r.t. a weight element."""
    layer = getattr(actor, layer_attr)
    W = layer.W
    grad_W = np.zeros_like(W)
    
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            W[i, j] += eps
            lp_plus = actor.log_prob(obs, act)
            W[i, j] -= 2 * eps
            lp_minus = actor.log_prob(obs, act)
            W[i, j] += eps  # restore
            grad_W[i, j] = (lp_plus - lp_minus) / (2 * eps)
    
    return grad_W


def analytical_gradient(actor, obs, act, adv=1.0):
    """Compute our analytical gradient for all three layers."""
    std = np.exp(actor.log_std)
    
    # Forward pass
    z1 = actor.fc1.forward(obs)
    h1 = relu(z1)
    z2 = actor.fc2.forward(h1)
    h2 = relu(z2)
    z_out = actor.mean_head.forward(h2)
    mean_val = 1.0 / (1.0 + np.exp(-z_out.flatten()))
    
    # d(log_prob)/d(mean)
    d_lp_d_mean = (act - mean_val[0]) / (std[0]**2)
    
    # sigmoid derivative
    sig_deriv = mean_val[0] * (1 - mean_val[0])
    
    # Signal at mean_head
    delta_out = d_lp_d_mean * sig_deriv  # * adv omitted (we compare d(log_prob), not d(loss))
    
    # mean_head gradient
    grad_mh = np.outer(h2, [delta_out])
    
    # fc2 gradient
    delta_h2 = delta_out * actor.mean_head.W.flatten()
    delta_z2 = delta_h2 * (z2 > 0).astype(np.float32)
    grad_fc2 = np.outer(h1, delta_z2)
    
    # fc1 gradient
    delta_h1 = delta_z2 @ actor.fc2.W.T
    delta_z1 = delta_h1 * (z1 > 0).astype(np.float32)
    grad_fc1 = np.outer(obs, delta_z1)
    
    return grad_fc1, grad_fc2, grad_mh


def main():
    print("=" * 60)
    print("  Chain Rule Verification: Analytical vs Finite-Difference")
    print("=" * 60)
    
    rng = np.random.RandomState(42)
    actor = MLPActor(rng)
    
    obs = np.array([0.5, 0.3, 0.0, 0.2], dtype=np.float32)
    act = 0.65
    
    # Analytical gradients
    grad_fc1_a, grad_fc2_a, grad_mh_a = analytical_gradient(actor, obs, act)
    
    # Numerical gradients
    print("\n  Computing finite-difference gradients (this takes a moment)...")
    grad_fc1_n = finite_diff_gradient(actor, obs, act, "fc1", "fc1")
    grad_fc2_n = finite_diff_gradient(actor, obs, act, "fc2", "fc2")
    grad_mh_n = finite_diff_gradient(actor, obs, act, "mean_head", "mean_head")
    
    # Compare
    for name, ga, gn in [("fc1", grad_fc1_a, grad_fc1_n),
                          ("fc2", grad_fc2_a, grad_fc2_n),
                          ("mean_head", grad_mh_a, grad_mh_n)]:
        max_diff = np.max(np.abs(ga - gn))
        rel_err = np.max(np.abs(ga - gn) / (np.abs(gn) + 1e-10))
        norm_a = np.linalg.norm(ga)
        norm_n = np.linalg.norm(gn)
        
        ok = max_diff < 1e-3
        status = "PASS" if ok else "FAIL"
        print(f"\n  {name}:")
        print(f"    Analytical norm:  {norm_a:.6f}")
        print(f"    Numerical norm:   {norm_n:.6f}")
        print(f"    Max abs diff:     {max_diff:.2e}")
        print(f"    Max rel error:    {rel_err:.2e}")
        print(f"    [{status}]")
        
        if not ok:
            print(f"    WARNING: Gradient mismatch detected!")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
