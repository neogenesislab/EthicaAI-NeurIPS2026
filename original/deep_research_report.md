# 계산 사회과학과 멀티 에이전트 강화학습을 통한 최선합리성(Optimal Rationality) 이론의 검증: 심층 연구 보고서

## 1. 서론: 합리성의 재구성과 계산적 전환

### 1.1 연구 배경: '합리적 바보'를 넘어서

고전 경제학 및 게임 이론의 중추를 이루는 합리적 선택 이론(Rational Choice Theory, RCT)은 인간의 행동을 일관된 선호 체계 내에서의 효용 극대화 과정으로 정의한다. 그러나 아마르티아 센(Amartya Sen)은 그의 기념비적인 논문 *Rational Fools*를 통해 이러한 정의가 인간 행위의 복잡성을 지나치게 단순화했음을 비판했다. 센에 따르면, 타인의 안녕을 자신의 효용 함수에 포함시키는 '동정(Sympathy)'과, 자신의 개인적 복리를 희생하면서까지 도덕적 원칙이나 사회적 규범을 따르는 '헌신(Commitment)'은 근본적으로 다른 기제이다. RCT는 이 두 가지를 모두 단일한 선호도(Preference)로 환원시킴으로써, 행위자가 자신의 이익을 추구하지 않는 상황조차도 "자신의 선호를 극대화하는 행위"라고 설명하는 동어반복적 오류를 범한다. 센은 이를 극복하기 위해 선호 자체에 대한 순위를 매기는 메타 랭킹(Meta-Rankings) 개념을 제안했다.

본 연구는 센의 철학적 통찰을 계산 사회과학(Computational Social Science, CSS) 및 멀티 에이전트 강화학습(MARL) 모델로 번역하여, 최선합리성(Optimal Rationality)이 실제 사회적 딜레마 상황에서 어떻게 발현되고 진화적 안정성을 갖는지 검증하는 것을 목표로 한다. 특히, 2024년 이후 급격히 발전한 JAX 기반의 고성능 시뮬레이션 환경을 활용하여, 기존의 소규모 게임 이론 모델이 포착하지 못했던 대규모 에이전트 집단 내의 규범 창발 현상을 규명하고자 한다.

### 1.2 연구의 필요성 및 목적

최근 AI 정렬(Alignment) 연구는 인공지능이 인간의 가치와 일치하도록 학습시키는 데 주력하고 있다. 그러나 '인간의 가치'가 단일한 보상 함수로 표현될 수 있다는 가정은 위험하다. 인간은 상황에 따라 이기적 본능(1차 선호)과 도덕적 의무(메타 선호) 사이에서 갈등하며 선택한다. 이러한 이중 구조를 AI 에이전트에 구현하지 않는다면, 우리는 센이 경고한 대로 '합리적 바보(Rational Fool)'—즉, 사회적 맥락을 무시하고 주어진 목적함수만을 맹목적으로 최적화하는 위험한 AI—를 양산할 가능성이 크다.

따라서 본 보고서는 다음의 네 가지 핵심 목표를 달성하기 위한 구체적인 로드맵을 제시한다:

1. **수학적 정식화**: 센의 메타 랭킹 이론을 강화학습의 보상 함수 및 정책 결정 구조로 변환한다.
2. **시뮬레이션 구현**: NVIDIA RTX 4070 (12GB VRAM) 환경 제약을 고려하여, 수천 명의 에이전트를 효율적으로 학습시킬 수 있는 JAX 기반 시뮬레이션 파이프라인을 설계한다.
3. **인과추론 검증**: 시뮬레이션 결과가 단순한 상관관계가 아님을 입증하기 위해 이중 기계학습(Double Machine Learning, DML)을 적용하여 '헌신' 기제의 인과적 효과(Causal Effect)를 추정한다.
4. **학술적 확산**: 독립 연구자로서 2025-2026년 주요 AI 및 사회과학 학회(NeurIPS, AIES, JASSS)에 연구 성과를 출판하기 위한 전략을 수립한다.

---

## 2. 최선합리성의 수학적 정식화 및 효용 함수 설계

최선합리성을 계산 모델로 구현하기 위해서는 철학적 개념을 수식으로 구체화해야 한다. 핵심은 에이전트가 단일한 보상($R$)을 최대화하는 것이 아니라, 상황($s$)에 따라 어떤 보상 함수를 따를지 결정하는 계층적 의사결정 구조를 갖는 것이다.

### 2.1 효용 함수의 계층적 분리

표준 강화학습(RL) 에이전트는 누적 보상의 기댓값 $J(\pi) = \mathbb{E}_{\tau \sim \pi} [\sum_t \gamma^t r(s_t, a_t)]$을 최대화한다. 센의 이론을 적용하기 위해, 우리는 보상을 **물질적 보상(Material Payoff)**과 **규범적 가치(Normative Value)**로 분리한다.

#### 2.1.1 1차 선호: 물질적 효용 ($U_{self}$)

환경으로부터 직접적으로 주어지는 보상(예: 자원 획득, 점수)이다. 이는 RCT의 '자기 이익(Self-Interest)'에 해당한다.

$$U_{self}(s, a) = r_{env}(s, a)$$

#### 2.1.2 메타 선호: 윤리적 평가 함수 ($U_{meta}$)

에이전트는 자신의 행동이 사회적 규범이나 집단 이익에 부합하는지를 평가하는 별도의 가치 함수를 가진다. 이는 센의 '헌신(Commitment)'을 나타낸다.

$$U_{meta}(s, a, \pi_{-i}) = \underbrace{\sum_{j \neq i} r_j(s, a)}_{\text{Sympathy}} + \underbrace{\mathbb{I}(a \in \mathcal{A}_{norm}) \cdot \omega}_{\text{Deontological Commitment}}$$

여기서 $\mathcal{A}_{norm}$은 규범적으로 허용된 행동 집합이며, $\omega$는 규범 준수의 가중치이다.

### 2.2 메타 랭킹 아키텍처 (Meta-Ranking Architecture)

단순히 $U_{self}$와 $U_{meta}$를 더하는 것은 센의 비판(모든 것을 하나의 효용으로 환원)을 벗어나지 못한다. 따라서 선택 메커니즘이 필요하다. 에이전트는 현재 상태 $s$에서 어떤 선호 체계($O$)를 활성화할지 결정하는 상위 정책 $\mu(O|s)$를 갖는다.

센의 이론을 반영한 **이중 프로세스 제어 루프(Dual-Process Control Loop)**는 다음과 같이 작동한다:

1. **상황 인식**: 에이전트는 상태 $s$를 관찰하고, 이것이 '사회적 딜레마' 상황인지 판단한다.
2. **메타 결정 (Meta-Decision)**: 메타 정책 $\mu$는 현재 상황에서 '이기적 모드($O_{self}$)'를 따를지 '헌신 모드($O_{commit}$)'를 따를지 확률적으로 결정한다. 이때, 헌신 모드의 선택 확률 $\lambda$는 에이전트의 '도덕적 성향'을 나타낸다.
3. **행동 선택**: 선택된 모드에 따라 결정된 최종 보상 함수 $R_{total}$을 기반으로 하위 정책 $\pi(a|s)$가 행동을 실행한다.

$$R_{total}(s, a) = (1 - \lambda_t) U_{self}(s, a) + \lambda_t [U_{meta}(s, a) - \psi(s)]$$

여기서 $\psi(s)$는 **자제 비용(Cost of Self-Control)**이다. 센은 헌신이 개인의 욕구를 억제하는 행위이므로 심리적 비용이 수반된다고 보았다. 이를 수식적으로 구현하기 위해, 본 연구에서는 에이전트의 현재 정책 분포와 본능적(이기적) 정책 분포 간의 쿨백-라이블러 발산(KL-Divergence)을 비용으로 도입한다.

$$\psi(s_t) = \beta \cdot D_{KL}(\pi(\cdot|s_t) || \pi_{selfish}(\cdot|s_t))$$

이 수식은 에이전트가 본능적 욕구($\pi_{selfish}$)에서 벗어난 행동을 할수록 더 높은 내적 비용을 치러야 함을 의미하며, 이는 진화적 관점에서 헌신적 행동이 왜 유지되기 어려운지, 그리고 그럼에도 불구하고 유지될 때 어떤 조건이 필요한지를 설명하는 핵심 변수가 된다.

### 2.3 사회적 딜레마 환경의 조건

이러한 수학적 모델을 검증하기 위해서는 환경이 순차적 사회적 딜레마(Sequential Social Dilemma, SSD)의 조건을 충족해야 한다.

1. **개인적 유인**: 모든 상태에서 배신(Defection)이 협력(Cooperation)보다 높은 즉각적 $U_{self}$를 제공한다.
2. **집단적 비효율**: 모두가 배신할 경우의 장기적 누적 보상은 모두가 협력할 경우보다 현저히 낮다.
3. **비정상성(Non-stationarity)**: 에이전트들의 정책이 학습에 따라 계속 변하므로, 고정된 최적 전략이 존재하지 않는다.

---

## 3. 시뮬레이션 프레임워크 비교 및 RTX 4070 최적화 전략

본 연구의 핵심 제약 사항은 단일 GPU(NVIDIA RTX 4070, 12GB VRAM) 환경에서 대규모 멀티 에이전트 시뮬레이션을 수행해야 한다는 점이다.

### 3.1 프레임워크 비교 분석

| 비교 항목 | PettingZoo (Standard) | OpenSpiel | SocialJax / JaxMARL (권장) |
|---|---|---|---|
| 실행 백엔드 | Python/NumPy (CPU) | C++ / Python (CPU) | JAX (GPU/TPU) |
| 병렬 처리 방식 | multiprocessing (오버헤드 큼) | 스레딩 (제한적) | vmap (자동 벡터화) |
| 데이터 전송 | CPU ↔ GPU 병목 발생 | CPU ↔ GPU 병목 발생 | VRAM 내부 처리 (Zero Copy) |
| 처리 속도 (SPS) | ~2,000 - 5,000 Steps/Sec | ~10,000 Steps/Sec | > 1,000,000 Steps/Sec |
| 메모리 관리 | 에이전트 별 객체 생성 (비효율) | 최적화됨 | XLA 컴파일로 정적 할당 |
| 사회적 딜레마 | 기본 환경 부족 | 게임 이론 중심 (Matrix Games) | SSD 특화 (Coin Game, Cleanup 등) |

**분석 결과**: PettingZoo나 OpenSpiel은 환경 시뮬레이션을 CPU에서 수행하므로, 수천 명의 에이전트가 상호작용할 때 CPU 연산이 병목이 되어 RTX 4070의 연산 능력을 10%도 활용하지 못하는 경우가 발생한다. 반면, **SocialJax (및 JaxMARL)**은 환경의 물리 엔진과 에이전트의 신경망 학습을 모두 JAX의 XLA(Accelerated Linear Algebra)로 컴파일하여 GPU 상에서 단일 그래프로 실행한다. 이는 최대 12,500배의 속도 향상을 가능케 하며, RTX 4070 한 대로 클러스터급 실험을 수행할 수 있게 해주는 유일한 대안이다.

### 3.2 RTX 4070 (12GB VRAM) 최적화 구현 전략

12GB VRAM은 대규모 딥러닝 모델에는 다소 부족할 수 있으나, MARL 시뮬레이션에서는 다음과 같은 전략을 통해 효율을 극대화할 수 있다.

1. **대규모 병렬 환경(Vectorized Environments)**: 단일 환경에 1,000명의 에이전트를 넣는 대신, 10명의 에이전트가 있는 환경을 **1,000개 병렬(Parallel Envs)**로 실행한다. JAX의 `jax.vmap`을 사용하면 1,000개의 환경을 동시에 스텝(step)시키는 연산이 행렬 곱셈처럼 처리되어 GPU 코어를 포화시킬 수 있다.

2. **파라미터 공유(Parameter Sharing)와 이질적 아키텍처**: 모든 에이전트가 개별적인 신경망을 가지면 VRAM이 부족하다. 대신 **동질적 파라미터 공유(Homogeneous Parameter Sharing)**를 적용하여, 모든 에이전트가 하나의 정책 네트워크($\pi_\theta$)를 공유하되, 입력으로 각자의 관측값($o_i$)과 에이전트 ID($id_i$)를 받도록 한다. 최선합리성 검증을 위해 네트워크를 두 그룹으로 나누어 그룹별 파라미터 공유를 적용한다.

3. **정밀도 혼합(Mixed Precision) 및 JIT 컴파일**: JAX의 `bfloat16` 또는 `float16`을 사용하여 정책 네트워크의 메모리 사용량을 절반으로 줄인다. 또한 `jax.jit`를 통해 전체 학습 루프를 컴파일하여 파이썬 오버헤드를 완전히 제거한다.

### 3.3 구현 코드 예시: SocialJax 내 메타 보상 주입

SocialJax 환경의 step 함수 내에 센의 메타 랭킹 로직을 주입하는 JAX 호환 의사 코드:

```python
import jax.numpy as jnp

def optimal_rationality_reward(agent_reward, group_reward, commitment_alpha, params):
    """
    최선합리성(Optimal Rationality) 이론에 기반한 보상 재조정 함수
    
    Args:
        agent_reward: 환경에서 받은 개인적 물질 보상 (Float)
        group_reward: 집단 전체의 보상 평균 또는 합계 (Float)
        commitment_alpha: 에이전트의 헌신 수준 (0.0 ~ 1.0, 학습 가능 파라미터)
        params: 자제 비용(Self-control cost) 계수 등
        
    Returns:
        shaped_reward: 메타 랭킹이 반영된 최종 효용
    """
    # 1. 헌신 수준에 따른 가중 평균 (Meta-Ranking Selection)
    utility = (1 - commitment_alpha) * agent_reward + \
              (commitment_alpha) * group_reward
              
    # 2. 자제 비용 (Cost of Commitment)
    deviance = jnp.abs(agent_reward - utility)
    control_cost = params['beta'] * deviance
    
    # 3. 생존 제약 (Survival Constraint)
    is_starving = agent_reward < params['survival_threshold']
    survival_penalty = jnp.where(is_starving, -100.0, 0.0)
    
    final_reward = utility - control_cost + survival_penalty
    return final_reward
```

---

## 4. 인과추론(Causal Inference)을 통한 이론 검증

단순히 시뮬레이션에서 협력이 발생했다고 해서 "최선합리성 이론이 맞다"고 결론 내릴 수는 없다. 환경적 요인(예: 풍부한 자원) 때문에 협력이 쉬웠을 수도 있기 때문이다. 따라서 **이중 기계학습(Double Machine Learning, DML)**을 사용하여 '헌신'이라는 처치(Treatment)가 '지속가능성'이라는 결과(Outcome)에 미치는 순수한 **인과적 효과(Causal Effect)**를 분리해내야 한다.

### 4.1 인과 모형 (Structural Causal Model) 설계

- **교란 변수 ($X$)**: 환경의 상태, 자원 재생성 속도, 에이전트 밀도 등 고차원적 상태 공간.
- **처치 ($T$)**: 에이전트들의 평균 헌신 수준 (Commitment Level, $\lambda$).
- **결과 ($Y$)**: 사회적 딜레마 해결 지표 (예: 자원 고갈까지 걸린 시간, 지니 계수).

### 4.2 이중 기계학습(DML) 검증 파이프라인

1. **데이터 수집**: SocialJax 시뮬레이션을 통해 $(X_i, T_i, Y_i)$ 궤적 데이터를 대량으로 수집한다.
2. **성가신 파라미터(Nuisance Parameter) 추정 (Stage 1)**:
   - 모델 $g(X)$: 환경 $X$가 결과 $Y$에 미치는 영향을 예측
   - 모델 $m(X)$: 환경 $X$가 처치 $T$에 미치는 영향을 예측
3. **잔차 계산 (Orthogonalization)**:
   - $\tilde{Y} = Y - g(X)$ (환경으로 설명되지 않는 결과의 변동)
   - $\tilde{T} = T - m(X)$ (환경으로 설명되지 않는 처치의 변동)
4. **인과 효과 추정 (Stage 2)**: 잔차 $\tilde{Y}$를 잔차 $\tilde{T}$에 대해 회귀분석한다. 이때 얻어지는 계수 $\theta$가 바로 헌신의 인과적 효과이다.

---

## 5. AI 정렬(Alignment)과의 연계 및 위험 관리

### 5.1 보상 해킹 방지와 'SanctSim' 메커니즘

에이전트에게 도덕적 보상을 주입할 때 가장 큰 위험은 에이전트가 실제로 유익한 행동을 하는 대신, 도덕적으로 보이는 신호만을 조작하여 보상을 탈취하는 것이다. 이를 방지하기 위해:

1. **감시자 에이전트(Auditor Agents)**: 학습 에이전트 외에 고정된 정책을 가진 감시자를 투입한다. 이들은 에이전트의 행동과 실제 환경 변화 간의 불일치를 감지하면 즉각적인 페널티(Sanction)를 부여한다.
2. **영향 정규화(Impact Regularization)**: 에이전트가 환경 상태를 급격하게 변화시키는 행위 자체에 대해 페널티를 부여함으로써, 불확실성이 높은 상황에서는 보수적으로 행동하도록 유도한다.

---

## 6. 시뮬레이션 테스트베드 설계 및 예상 결과

### 6.1 통합 자원 동역학 모델 (Unified Resource Dynamics)

세 가지 시나리오(어업, 목초지, 오염)는 모두 공통 자원(Common Pool Resource, CPR)의 차분 방정식으로 모델링된다:

$$h_{t+1} = \text{clip}(h_t - E_t + g \cdot (h_t - E_t), 0, h_{max})$$

- $h_t$: 현재 자원량 (물고기, 풀, 맑은 공기)
- $E_t = \sum a_{i,t}$: 모든 에이전트의 채취/오염 총량
- $g$: 자연 재생률 (Regeneration Rate)

### 6.2 실험 가설 및 시나리오

**가설**: 표준 PPO 에이전트(Rational Fools)는 관측 노이즈가 높을수록 상호 호혜성(Reciprocity) 전략이 붕괴되어 자원 고갈을 초래할 것이다. 반면, 메타 랭킹 에이전트(Optimal Rationality)는 즉각적인 보상 피드백과 무관하게 내재된 헌신 규범을 유지하므로, 노이즈가 높은 상황에서도 자원을 지속시킬 것이다.

---

## 7. 2024-2026 최신 문헌 및 학술 출판 전략

### 7.1 단계별 출판 로드맵

**Phase 1: 기술적 타당성 검증 (2025년 상반기)**
- 목표: 메타 랭킹 아키텍처가 대규모 시뮬레이션에서 기술적으로 작동함을 증명
- 타겟 학회: NeurIPS 2025 (Datasets & Benchmarks Track) 또는 Cooperative AI Workshop
- 핵심 성과물: JAX 기반의 OptimalRationalityEnv 오픈소스 공개

**Phase 2: 이론 및 인과 검증 (2026년 상반기)**
- 목표: 시뮬레이션 결과의 사회과학적 의미와 인과성 입증
- 타겟 학회/저널:
  - AIES (AAAI/ACM Conference on AI, Ethics, and Society)
  - JASSS (Journal of Artificial Societies and Social Simulation)
- 핵심 성과물: DML 분석을 포함한 심층 논문

### 7.2 독립 연구자(Independent Scholar)를 위한 전략

- **소속 표기**: "Independent Researcher"로 명기, ORCID 프로필 및 GitHub 관리
- **비용 지원(APC Waivers)**: ACM, PLOS, Royal Society 등의 논문 게재료 면제 프로그램 활용
- **사전 등록(Pre-registration)**: OSF (Open Science Framework)에 실험 계획을 사전 등록하여 투명성 확보

---

## 8. 결론

본 보고서는 아마르티아 센의 최선합리성 이론을 현대적인 계산 기법으로 검증하기 위한 포괄적인 가이드라인을 제시하였다. 우리는 (1) 헌신을 메타 랭킹 기반의 계층적 효용 함수로 정식화하고, (2) RTX 4070의 성능을 극대화할 수 있는 JAX 기반 SocialJax 프레임워크를 선정하였으며, (3) 인과추론(DML)을 통해 이론의 타당성을 통계적으로 엄밀하게 검증하는 방법을 설계하였다.

이 연구는 단순히 고전 경제학의 수정에 그치지 않고, 인간의 복잡한 도덕적 직관을 이해하고 이를 따를 수 있는 '안전하고 협력적인 AI'를 구축하는 데 필수적인 공학적 기반을 제공할 것이다.

### 주요 실행 과제 (Action Items)

1. **환경 구축**: JaxMARL 레포지토리를 포크(Fork)하여 `optimal_rationality_reward` 함수를 구현한다.
2. **베이스라인 확보**: 표준 PPO 에이전트로 '공유지의 비극'이 발생하는 붕괴 시점을 데이터로 확보한다.
3. **실험 수행**: 메타 랭킹 에이전트를 투입하여 자원 보존율의 변화를 기록하고, DML 파이프라인을 가동한다.
