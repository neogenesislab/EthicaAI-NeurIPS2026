"""
EthicaAI 시뮬레이션 설정 (Configuration)

연구 보고서의 파라미터와 시스템 설정을 중앙 관리.
Dataclass를 사용하여 타입 안정성 확보.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any

@dataclass
class RewardWeights:
    alpha: float = 0.4   # 개인 보상 (Self-Interest)
    beta: float = 0.3    # 사회적 보상 (Sympathy — 타인 보상 합)
    gamma: float = 0.2   # 윤리적 보상 (Commitment — 규범 준수)
    delta: float = 0.1   # 장기 위험 패널티

    def __post_init__(self) -> None:
        total = self.alpha + self.beta + self.gamma + self.delta
        # 부동소수점 오차 허용
        assert abs(total - 1.0) < 1e-6, (
            f"가중치 합이 1.0이어야 합니다 (현재: {total})"
        )

@dataclass
class MetaRankingParams:
    commitment_lambda: float = 0.5  # 초기 헌신 수준 (0.0=Selfish, 1.0=Altruistic)
    beta_kl: float = 0.1            # KL Divergence 가중치 (자제 비용)

@dataclass
class PrisonersDilemmaConfig:
    # 기본 보상: T > R > P > S
    # 2R > T+S (공동 협력이 낫다)
    reward_matrix: Dict[str, float] = field(default_factory=lambda: {
        "T": 5.0,  # Temptation (배신 vs 협력)
        "R": 3.0,  # Reward (협력 vs 협력)
        "P": 1.0,  # Punishment (배신 vs 배신)
        "S": 0.0   # Sucker (협력 vs 배신)
    })
    num_rounds: int = 200 # 테스트용으로 라운드 축소 (빠른 실행)
    noise_prob: float = 0.05 
    
    # 처벌(Sanction) 파라미터
    enable_sanction: bool = True
    punishment_cost: float = 1.0  # 처벌하는 자의 비용
    punishment_fine: float = 4.0  # 처벌받는 자의 벌금

@dataclass
class PublicGoodsConfig:
    multiplier: float = 1.6
    num_rounds: int = 100

@dataclass
class TrustGameConfig:
    multiplier: float = 3.0
    num_rounds: int = 100

@dataclass
class ResourceDynamicsConfig:
    initial_resource: float = 100.0
    depletion_rate: float = 0.01
    regeneration_rate: float = 0.02
    scarcity_threshold: float = 0.3

@dataclass
class AgentConfig:
    selfish_ratio: float = 0.4
    optimal_ratio: float = 0.4
    bounded_ratio: float = 0.2
    
    bounded_noise_std: float = 0.5 

@dataclass
class TrainConfig:
    algorithm: str = "PPO"
    learning_rate: float = 3e-4
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    
    use_fp16: bool = True
    num_envs: int = 4
    
    log_interval: int = 50 # 로그 자주 찍기

@dataclass
class CausalConfig:
    method: str = "DoubleML"
    treatment: str = "commitment_lambda"
    outcome: str = "cumulative_reward"
    confounders: List[str] = field(default_factory=lambda: ["resource_level", "opponent_type"])

@dataclass
class StatisticsConfig:
    significance_level: float = 0.05
    test_method: str = "t-test"

@dataclass
class ExperimentConfig:
    experiment_name: str = "EthicaAI_Sanction_Test"
    seed: int = 42
    output_dir: str = "simulation/outputs"
    
    reward_weights: RewardWeights = field(default_factory=RewardWeights)
    meta_ranking: MetaRankingParams = field(default_factory=MetaRankingParams)
    
    prisoners_dilemma: PrisonersDilemmaConfig = field(default_factory=PrisonersDilemmaConfig)
    public_goods: PublicGoodsConfig = field(default_factory=PublicGoodsConfig)
    trust_game: TrustGameConfig = field(default_factory=TrustGameConfig)
    
    resource: ResourceDynamicsConfig = field(default_factory=ResourceDynamicsConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    causal: CausalConfig = field(default_factory=CausalConfig)
    statistics: StatisticsConfig = field(default_factory=StatisticsConfig)

DEFAULT_CONFIG = ExperimentConfig()
