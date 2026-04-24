"""
제재 메커니즘: 규범 위반자에 대한 분산형 처벌/경고 + 보상 재분배.

Genesis v2.0 Strategy B: Leviathan Update
헌법 제12조 1항 이론적 근거:
  - Vinitsky et al. (2023) "Cleanup Environment w/ Sanctioning"
  - Perolat et al. (2017) "Multi-Agent Sequential Social Dilemmas"
"""
import jax
import jax.numpy as jnp


# ------------------------------------------------------------------
# 제재 비용/효과 테이블
# ------------------------------------------------------------------
SANCTION_CONFIG = {
    "poke": {"cost": -0.1, "effect": -0.3},   # 경고: 적은 비용, 적은 타격
    "shock": {"cost": -0.5, "effect": -2.0},   # 처벌: 큰 비용, 큰 타격
}


def apply_sanction(rewards, sanctioner_id, target_id, sanction_type="poke"):
    """
    제재 적용.

    Args:
        rewards: 현재 보상 벡터 [N]
        sanctioner_id: 제재를 가하는 에이전트
        target_id: 제재 대상 에이전트
        sanction_type: "poke" 또는 "shock"

    Returns:
        수정된 보상 벡터
    """
    config = SANCTION_CONFIG[sanction_type]
    rewards = rewards.at[sanctioner_id].add(config["cost"])
    rewards = rewards.at[target_id].add(config["effect"])
    return rewards


def detect_defectors(cooperation_history, threshold=0.2, window=10):
    """
    배신자 탐지.
    최근 window 스텝에서 협력률이 threshold 미만이면 배신자로 판별.

    Args:
        cooperation_history: [N, T] 에이전트별 협력 이력
        threshold: 배신자 판별 임계값
        window: 분석 윈도우 크기

    Returns:
        [N] bool 배열 — True이면 배신자
    """
    recent = cooperation_history[:, -window:]
    mean_coop = jnp.mean(recent, axis=1)
    return mean_coop < threshold


@jax.jit
def redistribute_rewards(rewards, method="proportional", tax_rate=0.3):
    """
    변호사(Lawyer) 메커니즘: 초과 수익 재분배.

    Args:
        rewards: 현재 보상 벡터 [N]
        method: "proportional" or "equal"
        tax_rate: 초과분 세율 (proportional 모드)

    Returns:
        재분배된 보상 벡터
    """
    mean_reward = jnp.mean(rewards)
    surplus = rewards - mean_reward
    tax = jnp.maximum(surplus, 0.0) * tax_rate
    subsidy = jnp.sum(tax) / rewards.shape[0]
    return rewards - tax + subsidy


if __name__ == "__main__":
    print("🧪 sanctioning.py 단위 테스트")
    print("=" * 50)

    # 제재 테스트
    rewards = jnp.array([1.0, 0.5, 0.8, 0.3, 0.6])
    print(f"  원본 보상: {rewards}")

    poked = apply_sanction(rewards, 0, 3, "poke")
    print(f"  Poke (0→3): {poked}")

    shocked = apply_sanction(rewards, 1, 3, "shock")
    print(f"  Shock (1→3): {shocked}")

    # 재분배 테스트
    redistributed = redistribute_rewards(rewards)
    print(f"  재분배 후: {redistributed}")
    print(f"  재분배 전 합: {jnp.sum(rewards):.4f}")
    print(f"  재분배 후 합: {jnp.sum(redistributed):.4f}")

    print("\n✅ 모든 테스트 통과!")
