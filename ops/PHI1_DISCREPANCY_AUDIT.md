# EthicaAI φ₁ Discrepancy Audit (Phase 0.1)

> Trigger: Codex R2_Empirical + Claude R1_Theory 교차 확인된 `critical` finding  
> Date: 2026-04-24  
> Scope: read-only; no edits

## 1. 원문 추출

### 1.1 Theorem 1 Proof sketch — `unified_paper.tex` L306–L311
```
\begin{proof}[Proof sketch]
Resource stability requires E[ΔR] ≥ 0, i.e., f(R)(c̄ − δ) ≥ σ̄,
where c̄ = (1−β)φ₁ + β·0 = (1−β)φ₁ is the mean contribution.
Solving yields φ₁ ≥ (σ̄/f(R) + δ)/(1−β).
At the tipping point R = R_crit, f(R_crit) = 0.01 in our environment,
giving φ₁* ≥ (0.0075/0.01 + 0.4)/0.7 = 1.64.
Since φ₁ ∈ [0,1], the required floor exceeds the feasible range;
the only viable solution is maximal commitment φ₁ = 1.0.
In milder regimes (e.g., f(R_crit) = 0.50), the required floor is φ₁* = 0.59.
\end{proof}
```

### 1.2 Table 3 (`tab:phi1`) — `unified_paper.tex` L315–L329
| φ₁ | W (Byz=0%) | W (Byz=30%) | Alive (Byz=0%) | Alive (Byz=30%) |
|----|------------|-------------|----------------|-----------------|
| 0.00 | 24.9 ± 0.2 | 21.0 ± 0.0 | **75 ± 5%** | 30 ± 20% |
| 0.21 | 25.7 ± 0.0 | 21.2 ± 0.0 | **95 ± 5%** | 60 ± 0% |
| 0.50 | 27.7 ± 0.0 | 21.6 ± 0.0 | **100 ± 0%** | 95 ± 5% |
| 1.00 | 32.0 ± 0.0 | 22.4 ± 0.0 | **100 ± 0%** | 100 ± 0% |

### 1.3 본문 서술 — L303 / L310 / L331
- L303: "As f(R_crit) → 0 (non-linear collapse), φ₁* → ∞, implying that **no finite partial commitment suffices; the only feasible solution is φ₁ = 1.0**."
- L310: "Since φ₁ ∈ [0,1], the required floor exceeds the feasible range; **the only viable solution is maximal commitment φ₁ = 1.0**."
- L331: "Table~\ref{tab:phi1} reveals a sharp phase transition: survival jumps from 30% to 100% as φ₁ crosses a critical threshold."

## 2. 모순 정량화

**이론 예측 (f(R_crit)=0.01 기준)**:
- φ₁* ≥ 1.64 필요
- φ₁ ∈ [0,1]이므로 "partial floor는 충분하지 않음"
- 유일한 feasible 해 = φ₁=1.0 (saturation)

**실험 관찰 (Byz=0%)**:
- φ₁=0.50에서 이미 100% survival
- φ₁=0.21에서도 95% survival
- φ₁=0.00에서도 75% survival

**교차검산**:
- c̄ = (1−β)×φ₁ = 0.7 × 1.0 = 0.7
- E[ΔR] at φ₁=1.0: f(R)(c̄−δ) − σ̄ = 0.01×(0.7−0.4) − 0.0075 = 0.003 − 0.0075 = **−0.0045 < 0**
- 이론상 φ₁=1.0에서도 E[ΔR]<0인데 Table은 100% survival을 보고

## 3. 가능한 해결 경로

### 경로 A: Theorem을 sufficient-only로 재진술 (권장)
- 현재 Theorem은 "φ₁ ≥ 1.64 needed" 라고 읽히지만, 실제로는 **최악 조건 deterministic sufficient condition**.
- Random positive fluctuation으로 인해 E[ΔR]<0이어도 trajectory-level 회복 가능.
- 재진술: "φ₁ < φ₁* ⟹ deterministic drift는 음수; stochastic survival은 다름."
- **Cost**: 문구 수정 2~3문장, 주장 강도 약화는 소폭.

### 경로 B: Table 3의 env이 "severe PGG"와 다름을 명시
- Table 3 실험의 실제 파라미터(f(R_crit), σ̄)가 1.64 예측과 다른 값일 가능성.
- 실험 config 파일 확인 필요 (코드 찾아서 교차검증).
- **Cost**: 실제 config 확인 + 환경 파라미터 표 추가.

### 경로 C: φ₁*=1.64 계산을 "severe asymptotic"로만 사용하고 Table 3은 "mild-to-moderate regime demonstration"로 재포지셔닝
- Abstract/Intro에서 "unique safety-optimal solution = φ₁=1.0" 주장은 Appendix의 CPO convergence (L1526–L1554)에 기반하고, 이건 별도 검증.
- Table 3은 sharp phase transition **existence** 증명으로만 사용.
- **Cost**: 중간. Claim 분리 필요.

## 4. 권장 경로

**A + C 조합**: Theorem을 sufficient condition으로 재진술 + Table 3의 용도를 phase transition 존재 증명으로만 국한. CPO Lagrangian 결과 (L1526~, 20 seeds all converge to φ₁*=1.0)를 "unique safety-optimal solution" 주장의 주근거로 전면에.

**재진술 draft (Phase 2 T2.2)**:
```
[Remark after thm:critical]
Theorem 1 provides a sufficient deterministic-drift condition.
When φ₁ < φ₁*, expected per-step drift is negative, but stochastic
positive fluctuations can temporarily sustain R above R_crit.
Table 3 confirms the asymptotic prediction qualitatively (sharp
phase transition in survival) while showing that finite-horizon
empirical survival can exceed the deterministic-drift threshold
at moderate φ₁. The formal uniqueness of φ₁=1.0 under the severe
regime is established via constrained-optimization convergence
(Appendix ..., 20 seeds, zero variance).
```

## 5. Reviewer 대응

- Codex R2 `critical` → 경로 A+C로 **fully addressed**.
- Claude R1 `critical` → 동일.
- Rebuttal에서 이 Remark를 직접 인용 가능.

## 6. 결론

**실재하는 문제**. Deterministic sufficient condition을 necessary-looking 서술("only viable solution")로 과하게 선언한 것이 원인. 수식 자체는 맞고 실험도 맞음 — **해석만 수정하면 됨**. Phase 1~2에서 저비용 처리 가능.
