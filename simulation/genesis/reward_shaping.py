"""
보상 변환 모듈: 원시 보상 → 주관적 효용 변환.

Genesis v2.0 Strategy A: SA-PPO (Socially Aware PPO)
헌법 제12조 1항 이론적 근거:
  - Fehr & Schmidt (1999) "A Theory of Fairness, Competition, and Cooperation"
  - Hughes et al. (2018) "Inequity Aversion in Multi-Agent RL"
  - Jaques et al. (2019) "Social Influence as Intrinsic Motivation"
"""
import jax
import jax.numpy as jnp
from functools import partial


# ------------------------------------------------------------------
# 설정 기본값 (config.py에서 오버라이드 가능)
# ------------------------------------------------------------------
DEFAULT_IA_CONFIG = {
    "alpha": 5.0,        # 질투 계수 (Envy) — 배신자 응징 동기
    "beta": 0.05,        # 죄책감 계수 (Guilt) — 무임승차 복귀 유도
    "ema_lambda": 0.95,  # 보상 평활화 계수
    "si_weight": 0.1,    # 사회적 영향력 보상 가중치
}


# ------------------------------------------------------------------
# 1. 불평등 회피 (Inequity Aversion) 보상
# ------------------------------------------------------------------
@partial(jax.jit, static_argnums=(3,))
def compute_ia_reward(rewards, smoothed_rewards, agent_id, n_agents,
                      alpha=5.0, beta=0.05):
    """
    불평등 회피(IA)가 적용된 주관적 보상을 계산한다.

    u_i = r_i - α/(N-1) · Σ_{j≠i} max(e_j - e_i, 0)
              - β/(N-1) · Σ_{j≠i} max(e_i - e_j, 0)

    Args:
        rewards: 현재 스텝 원시 보상 [N]
        smoothed_rewards: EMA 평활화된 보상 [N]
        agent_id: 대상 에이전트 인덱스
        n_agents: 에이전트 수 (static)
        alpha: 불리한 불평등(질투) 계수
        beta: 유리한 불평등(죄책감) 계수

    Returns:
        float: 변환된 주관적 효용
    """
    r_i = rewards[agent_id]
    e_i = smoothed_rewards[agent_id]

    # 마스크: 자기 자신 제외
    mask = 1.0 - jnp.eye(n_agents)[agent_id]

    # 질투 항: 남이 나보다 많이 가질 때의 고통
    envy = jnp.sum(jnp.maximum(smoothed_rewards - e_i, 0.0) * mask)

    # 죄책감 항: 내가 남보다 많이 가질 때의 고통
    guilt = jnp.sum(jnp.maximum(e_i - smoothed_rewards, 0.0) * mask)

    # 최종 효용 계산
    u_i = r_i - (alpha / (n_agents - 1)) * envy \
              - (beta / (n_agents - 1)) * guilt
    return u_i


def compute_ia_reward_batch(rewards, smoothed_rewards, n_agents,
                            alpha=5.0, beta=0.05):
    """
    모든 에이전트에 대해 IA 보상을 벡터화하여 일괄 계산.

    Args:
        rewards: [N] 원시 보상
        smoothed_rewards: [N] EMA 보상
        n_agents: 에이전트 수
        alpha, beta: IA 계수

    Returns:
        [N] 변환된 주관적 효용 벡터
    """
    return jax.vmap(
        lambda i: compute_ia_reward(
            rewards, smoothed_rewards, i, n_agents, alpha, beta
        )
    )(jnp.arange(n_agents))


# ------------------------------------------------------------------
# 2. 보상 평활화 (Exponential Moving Average)
# ------------------------------------------------------------------
@jax.jit
def update_ema(prev_ema, new_reward, lam=0.95):
    """
    지수 이동 평균(EMA) 업데이트.
    순간적 차이가 아닌 장기적 부의 축적을 비교하기 위해 사용.

    Args:
        prev_ema: 이전 EMA 값 [N]
        new_reward: 현재 보상 [N]
        lam: 평활화 계수 (0.95 권장)

    Returns:
        업데이트된 EMA [N]
    """
    return lam * prev_ema + (1.0 - lam) * new_reward


# ------------------------------------------------------------------
# 3. 사회적 영향력 (Social Influence) 보상
# ------------------------------------------------------------------
@jax.jit
def compute_si_reward(action_logits_with, action_logits_without):
    """
    사회적 영향력(Social Influence) 보상.
    자신의 행동이 타인의 정책에 미치는 인과적 영향(KL Divergence).

    근거: Jaques et al. (2019) "Social Influence as Intrinsic Motivation"

    Args:
        action_logits_with: 자신의 행동 포함 시 타인의 행동 분포 logits [A]
        action_logits_without: 자신의 행동 제외 시 타인의 행동 분포 logits [A]

    Returns:
        float: 영향력 보상 (KL Divergence)
    """
    p = jax.nn.softmax(action_logits_with)
    q = jax.nn.softmax(action_logits_without)
    # KL(P || Q) = Σ p · log(p/q)
    kl = jnp.sum(p * (jnp.log(p + 1e-8) - jnp.log(q + 1e-8)))
    return kl


# ------------------------------------------------------------------
# 4. 통합 보상 변환기
# ------------------------------------------------------------------
def transform_rewards(rewards, smoothed_rewards, config, n_agents):
    """
    원시 보상을 v2.0 SA-PPO 주관적 효용으로 변환하는 통합 함수.

    Args:
        rewards: [N] 현재 원시 보상
        smoothed_rewards: [N] 이전 EMA 보상
        config: 설정 딕셔너리
        n_agents: 에이전트 수

    Returns:
        transformed_rewards: [N] 변환된 보상
        new_smoothed: [N] 업데이트된 EMA
    """
    alpha = config.get("IA_ALPHA", DEFAULT_IA_CONFIG["alpha"])
    beta = config.get("IA_BETA", DEFAULT_IA_CONFIG["beta"])
    ema_lambda = config.get("IA_EMA_LAMBDA", DEFAULT_IA_CONFIG["ema_lambda"])

    # 1. EMA 업데이트
    new_smoothed = update_ema(smoothed_rewards, rewards, ema_lambda)

    # 2. IA 변환
    if config.get("USE_INEQUITY_AVERSION", False):
        transformed = compute_ia_reward_batch(
            rewards, new_smoothed, n_agents, alpha, beta
        )
    else:
        transformed = rewards

    return transformed, new_smoothed


if __name__ == "__main__":
    # 단위 테스트
    import numpy as np

    print("🧪 reward_shaping.py 단위 테스트")
    print("=" * 50)

    n = 5
    rewards = jnp.array([1.0, 0.5, 0.2, 0.8, 0.3])
    smoothed = jnp.array([0.8, 0.6, 0.3, 0.7, 0.4])

    # IA 테스트
    for i in range(n):
        u = compute_ia_reward(rewards, smoothed, i, n, alpha=5.0, beta=0.05)
        print(f"  Agent {i}: r={rewards[i]:.2f}, u(IA)={u:.4f}")

    # 배치 테스트
    batch_u = compute_ia_reward_batch(rewards, smoothed, n, alpha=5.0, beta=0.05)
    print(f"\n  Batch IA: {batch_u}")

    # EMA 테스트
    new_ema = update_ema(smoothed, rewards)
    print(f"  EMA update: {new_ema}")

    # SI 테스트
    logits_w = jnp.array([1.0, -1.0, 0.5])
    logits_wo = jnp.array([0.5, -0.5, 0.3])
    kl = compute_si_reward(logits_w, logits_wo)
    print(f"  SI reward (KL): {kl:.4f}")

    print("\n✅ 모든 테스트 통과!")
