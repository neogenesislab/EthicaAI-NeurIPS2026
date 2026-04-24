"""
EthicaAI Experiment Configurations
Defines hyperparameters for small, medium, and large-scale experiments.
All tunable parameters are centralized here (하드코딩 금지 원칙).
"""
import math

# --- HRL Shared Defaults ---
_HRL_DEFAULTS = {
    "HRL_NUM_NEEDS": 2,
    "HRL_NUM_TASKS": 2,
    "HRL_ALPHA": 1.0,
    "HRL_THRESH_INCREASE": 0.005,
    "HRL_THRESH_DECREASE": 0.05,
    "HRL_INTAKE_VAL": 0.2,
    # Environment Reward Scale (Solution A)
    "REWARD_APPLE": 10.0,
    "COST_BEAM": -1.0,
    # Meta-Ranking Parameters (Sen's Optimal Rationality)
    "META_BETA": 0.1,                # ψ self-control cost coefficient
    "META_SURVIVAL_THRESHOLD": -5.0,  # wealth below → λ→0 (survival mode)
    "META_WEALTH_BOOST": 5.0,         # wealth above → λ×1.5 (generosity)
    "META_LAMBDA_EMA": 0.9,           # λ smoothing factor (EMA)
    # Dynamic λ resource thresholds — Eq. (2) in paper
    "R_CRISIS": 0.2,                   # Resource crisis threshold (survival mode)
    "R_ABUNDANCE": 0.7,                # Resource abundance threshold (generosity mode)
    # Experiment Control Flags
    "USE_META_RANKING": True,          # False = baseline (순수 SVO 보상변환)
    "META_USE_DYNAMIC_LAMBDA": True,   # False = λ = sin(θ) 고정 (ablation)
    # Genesis Autonomous Research Flags
    "GENESIS_MODE": False,             # True = Genesis 실험 모드 활성화
    "GENESIS_BETA_BASE": 10.0,        # Genesis: beta 기본값
    "GENESIS_GAMMA": 2.0,             # Genesis: 감쇠 계수
    "GENESIS_ALPHA": 5.0,             # Genesis: 증폭 계수
    # Infrastructure
    "DASHBOARD_PORT": 4011,            # Local port allocation (4011~4020)
    # v2.0: Inequity Aversion (SA-PPO) 파라미터
    "USE_INEQUITY_AVERSION": True,     # SA-PPO 활성화
    "IA_ALPHA": 5.0,                   # 질투 계수 (Envy) — Fehr & Schmidt (1999)
    "IA_BETA": 0.05,                   # 죄책감 계수 (Guilt)
    "IA_EMA_LAMBDA": 0.95,             # 보상 평활화 계수
    "SI_WEIGHT": 0.1,                  # 사회적 영향력 보상 가중치
    # v2.0: Mediator 파라미터
    "MEDIATOR_K": 10,                  # 위임 결정 주기
    "MEDIATOR_LAMBDA_IC": 1.0,         # IC 제약 강도
    "MEDIATOR_LAMBDA_E": 0.5,          # E 제약 강도
}

# --- Small Scale (Development & Debugging) ---
CONFIG_SMALL = {
    "ENV_NAME": "cleanup",
    "NUM_AGENTS": 5,
    "ENV_HEIGHT": 25,
    "ENV_WIDTH": 18,
    "MAX_STEPS": 100,
    "NUM_ENVS": 4,
    "NUM_UPDATES": 2,
    "ROLLOUT_LEN": 16,
    "BATCH_SIZE": 4 * 16,
    "LR": 2.5e-4,
    "HIDDEN_DIM": 128,
    "GAMMA": 0.99,
    "GAE_LAMBDA": 0.95,
    "CLIP_EPS": 0.2,
    "ENTROPY_COEFF": 0.01,
    "VF_COEFF": 0.5,
    "MAX_GRAD_NORM": 0.5,
    **_HRL_DEFAULTS,
}

# --- Medium Scale (Standard Experiments) ---
CONFIG_MEDIUM = {
    "ENV_NAME": "cleanup",
    "NUM_AGENTS": 20,
    "ENV_HEIGHT": 36,
    "ENV_WIDTH": 25,
    "MAX_STEPS": 500,
    "NUM_ENVS": 16,
    "NUM_UPDATES": 300,  # 100→300 for meta-ranking policy differentiation
    "ROLLOUT_LEN": 128,
    "BATCH_SIZE": 16 * 128,
    "LR": 3e-4,
    "HIDDEN_DIM": 128,
    "GAMMA": 0.99,
    "GAE_LAMBDA": 0.95,
    "CLIP_EPS": 0.2,
    "ENTROPY_COEFF": 0.05,  # Solution D: exploration boost
    "VF_COEFF": 0.5,
    "MAX_GRAD_NORM": 0.5,
    **_HRL_DEFAULTS,
}

# --- Large Scale (Research-Grade) ---
CONFIG_LARGE = {
    "ENV_NAME": "cleanup",
    "NUM_AGENTS": 100,
    "ENV_HEIGHT": 50,
    "ENV_WIDTH": 50,
    "MAX_STEPS": 500,
    "NUM_ENVS": 8,      # 16 → 8 (OOM 방지, 12GB VRAM)
    "NUM_UPDATES": 300,  # 100 → 300 (충분한 학습, Medium과 동일)
    "ROLLOUT_LEN": 128,
    "BATCH_SIZE": 8 * 128,
    "LR": 3e-4,
    "HIDDEN_DIM": 256,
    "GAMMA": 0.99,
    "GAE_LAMBDA": 0.95,
    "CLIP_EPS": 0.2,
    "ENTROPY_COEFF": 0.01,
    "VF_COEFF": 0.5,
    "MAX_GRAD_NORM": 0.5,
    **_HRL_DEFAULTS,
}

# --- Large Scale Harvest (일반화 검증) ---
CONFIG_LARGE_HARVEST = {
    **CONFIG_LARGE,
    "ENV_NAME": "harvest",  # Cleanup → Harvest 환경
}

# --- SVO Sweep Configurations ---
SVO_SWEEP_THETAS = {
    "selfish": 0.0,                    # 0°
    "individualist": math.pi / 12,     # 15°
    "competitive": math.pi / 6,        # 30°
    "prosocial": math.pi / 4,          # 45° (π/4)
    "cooperative": math.pi / 3,        # 60°
    "altruistic": 5 * math.pi / 12,    # 75°
    "full_altruist": math.pi / 2,      # 90° (π/2)
}

# Utility
def get_config(scale="small"):
    configs = {
        "small": CONFIG_SMALL,
        "medium": CONFIG_MEDIUM,
        "large": CONFIG_LARGE,
        "large_harvest": CONFIG_LARGE_HARVEST,
    }
    return configs.get(scale, CONFIG_SMALL)
