"""
Genesis Experimental Extension for train_pipeline.py
=====================================================
This file contains experimental modifications to the Meta-Ranking logic
for the Genesis (Self-Evolving Research Ecosystem) project.

Usage:
  Import this module and call `apply_genesis_reward()` instead of the
  standard meta-ranking reward calculation in train_pipeline.py.

  To activate, set config["GENESIS_MODE"] = True in config.py.

WARNING: These are EXPERIMENTAL hypotheses. Do NOT merge into the main
         train_pipeline.py without thorough validation.
"""
import jax.numpy as jnp


def apply_genesis_reward(combined_rewards, r_avg_others, lambda_base,
                         flat_levels, config, mode="inverse_beta"):
    """
    Genesis Meta-Ranking 보상 계산 (실험용).

    Args:
        combined_rewards: (B, N) 개별 에이전트 보상
        r_avg_others: (B, N) 타인 평균 보상
        lambda_base: scalar, sin(svo_theta)
        flat_levels: (B*N, NumNeeds) HRL 내부 상태
        config: dict, 실험 설정
        mode: "adaptive_beta" | "inverse_beta" | "institutional"

    Returns:
        meta_ranking_rewards: (B, N)
        debug_info: dict (beta_t, instability 등 디버깅용)
    """
    B, N = combined_rewards.shape

    # 공통: Instability 계산
    reward_gap = jnp.abs(r_avg_others - combined_rewards)  # (B, N)
    instability = reward_gap.mean(axis=-1, keepdims=True)   # (B, 1)

    # Common: Wealth Calculation
    wealth = flat_levels.mean(axis=-1).reshape((B, N))  # (B, N)

    # --- Hypothesis #001: Adaptive Beta (감쇠형) ---
    if mode == "adaptive_beta":
        beta_base = config.get("GENESIS_BETA_BASE", 10.0)
        gamma = config.get("GENESIS_GAMMA", 2.0)
        beta_t = beta_base * jnp.exp(-gamma * instability)

        # beta_t를 lambda 스케일링에 적용
        adaptive_scale = jnp.exp(-gamma * instability)
        lambda_dynamic = lambda_base * adaptive_scale

    # --- Hypothesis #002: Inverse Adaptive Beta (증폭형 - Forced Cooperation) ---
    elif mode == "inverse_beta":
        beta_base = config.get("GENESIS_BETA_BASE", 10.0)
        alpha = config.get("GENESIS_ALPHA", 5.0)
        
        # Instability가 높을수록 Beta가 급격히 커짐
        beta_t = beta_base * (1.0 + alpha * instability)

        # 핵심: 위기 상황(High Beta)에서는 본성(lambda_base)을 무시하고 강제 협력(1.0) 유도
        # Lambda = Base + (1 - Base) * tanh(Beta * Instability)
        forced_signal = jnp.tanh(beta_t * instability)
        lambda_dynamic = lambda_base + (1.0 - lambda_base) * forced_signal

    # --- Hypothesis #003: Institutional Punishment (Real-World Sanctions) ---
    elif mode == "institutional":
        # Mechanism: Sanction on Unfair Gain (부당 이득 환수)
        # "법을 어기면(이기적으로 이득을 취하면) 처벌받는다"
        # GENESIS_BETA = Penalty Rate (처벌 강도)
        
        penalty_rate = config.get("GENESIS_BETA", 1.0)
        
        # Selfish Gain: 나만 잘나서 얻은 이득 (Defection Advantage)
        # (My Reward - Avg Others)가 양수이면 이기적인 상태
        selfish_gain = jnp.maximum(0.0, combined_rewards - r_avg_others)
        
        # Sanction: 부당 이득에 비례한 벌금
        # 부자일수록(Wealthy) 더 강하게 처벌? (User preference: 'Real society')
        # Here we apply strict penalty on the ACT of selfishness.
        sanction = penalty_rate * selfish_gain
        
        # Lambda is NOT forced. Agents must LEARN to avoid Sanction.
        lambda_dynamic = lambda_base
        beta_t = jnp.ones((B, 1)) * penalty_rate

    else:
        # Fallback
        beta_t = jnp.zeros((B, 1))
        lambda_dynamic = lambda_base
        sanction = 0.0

    # Wealth-based Dynamic Lambda (기존 로직 유지 - Secondary Effect)
    # 생존 본능은 그대로 둠 (배고프면 이기적이 됨)
    meta_survival = config.get("META_SURVIVAL_THRESHOLD", -5.0)
    meta_boost = config.get("META_WEALTH_BOOST", 5.0)

    lambda_dynamic = jnp.where(
        wealth < meta_survival,
        0.0,
        jnp.where(
            wealth > meta_boost,
            jnp.minimum(1.0, lambda_dynamic * 1.5),
            lambda_dynamic
        )
    )

    # Psi (Self-control cost)
    meta_beta = config.get("META_BETA", 0.1)
    psi = meta_beta * reward_gap

    # Final Reward Mixing
    # Reward = (Intrinsic Preference) - (Institutional Sanction)
    base_reward = (1 - lambda_dynamic) * combined_rewards + \
                  lambda_dynamic * (r_avg_others - psi)
    
    # Apply Sanction only in Institutional Mode
    if mode == "institutional":
        meta_ranking_rewards = base_reward - sanction
    else:
        meta_ranking_rewards = base_reward

    debug_info = {
        "beta_t": beta_t,
        "instability": instability,
        "lambda_dynamic": lambda_dynamic,
        "reward_gap": reward_gap,
    }

    return meta_ranking_rewards, debug_info
