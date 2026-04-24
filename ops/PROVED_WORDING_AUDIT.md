# EthicaAI "proved" vs "conditional" Wording Audit (Phase 0.2)

> Trigger: Claude R1_Theory `critical` — thm:poa 헤더에서 "conditional on Conjecture"로 읽히는데 Abstract/Intro는 "proved"로 서술함
> Date: 2026-04-24
> Scope: read-only; no edits

## 1. 원문 재확인

### 1.1 Abstract (`unified_paper.tex` L93–L94)
```
\textbf{Theory (proved).}~Signal dilution (O(ε/N)) and basin dominance (measure ≥ 1−e^{−cN}) together
establish PoA ≥ Ω(N)---polynomial divergence contrasting sharply with O(1) in standard social dilemmas.
\textbf{Theory (conjectured).}~Escape may require Ω(e^{cN}) gradient evaluations;
we state this as a conjecture, not a proved theorem.
```

### 1.2 Introduction §1.2 (L118–L123)
```
We prove two mechanisms (Proposition thm:impossibility):
(i) signal dilution: O(ε/N); (ii) basin dominance: measure ≥ 1−e^{−cN}.
These proved results establish our main theoretical contribution: PoA ≥ Ω(N) (Theorem thm:poa)
---polynomial divergence not seen in standard social dilemmas (PoA = O(1)).
We additionally present empirical evidence and Freidlin--Wentzell-style heuristics suggesting that
escape may require Ω(e^{cN}) gradient evaluations; we record this stronger scaling as
Conjecture conj:escape, not as a proved theorem. Conditional on this conjecture, PoA strengthens to Ω(e^{cN}).
```

### 1.3 Theorem thm:poa 헤더 (L344–L347)
```
\begin{theorem}[Price of Anarchy Divergence in TPSDs
  {\normalfont (conditional on Conjecture~\ref{conj:escape})}]
\label{thm:poa}
In any TPSD with N agents and Byzantine fraction β < 1−δ*/φ1*,
**if Conjecture conj:escape** (Part iii) holds, then PoA ≥ Ω(e^{cN}) for c>0, vs O(1) in standard social dilemmas.
*Without the conjecture, Parts (i)--(ii) alone establish PoA ≥ Ω(N) (polynomial divergence).*
\end{theorem}
```

### 1.4 Intractability remark (L342)
```
\emph{Note}: Parts (i)–(ii) are rigorous and self-contained; the PoA divergence ≥ Ω(N) follows from these alone
(see Theorem thm:poa below). Part (iii) would strengthen this to Ω(e^{cN}) but is not required for the core results.
```

## 2. 구조 분석

논리적 주장은 실제로는 **정확**하다:
- Parts (i) + (ii)만으로 Ω(N) polynomial divergence는 **proved**
- Conjecture conj:escape를 추가하면 Ω(e^{cN})으로 강화 (conditional)

Abstract와 Intro와 Intractability remark는 모두 이 두 층을 **명시적으로 구분**해서 서술한다:
- Abstract: "Theory (proved)" ↔ "Theory (conjectured)"
- Intro: "These proved results establish our main theoretical contribution: PoA ≥ Ω(N)" ↔ "Conditional on this conjecture, PoA strengthens to Ω(e^{cN})"
- L342 remark: "Parts (i)–(ii) are rigorous and self-contained"

**문제는 Theorem 헤더 단독으로 읽을 때 일관성이 깨진다는 것**:
- 헤더 subtitle = "(conditional on Conjecture conj:escape)"
- 본문 첫 문장 = "if Conjecture holds, then PoA ≥ Ω(e^{cN})"
- 본문 마지막 문장 = "Without the conjecture, Parts (i)–(ii) alone establish PoA ≥ Ω(N)"

즉 **Theorem 본문에 두 개의 결과가 섞여 있는데 헤더는 조건부 결과만 가리킨다**. 리뷰어가 Theorem만 스캔할 경우 "모든 주장이 conditional"로 오독 가능.

## 3. 리뷰어 우려의 실체

Claude R1_Theory가 `critical`로 본 이유는 아마 다음 중 하나:

**가설 A (오독)**: 리뷰어가 Theorem 헤더의 "(conditional)"만 보고 Abstract의 "proved"와 충돌한다고 판단.
→ 실제로는 Abstract가 맞고 (Ω(N) is proved), Theorem 헤더가 overly restrictive하게 명명됨.

**가설 B (본질적 문제)**: Theorem이라는 형식적 객체 안에 두 개의 결과(proved Ω(N) + conditional Ω(e^{cN}))가 들어가 있는 것 자체가 비표준적.
→ 통상 theorem 1개 = claim 1개. "Without the conjecture" 같은 양자택일 분기가 theorem statement에 들어가는 건 수학 관행에서 드물다.

두 가설 모두 **wording/framing 문제**이고 **수학적 오류는 아니다**.

## 4. 해결 경로

### 경로 A: Theorem 분할 (권장)
- `\begin{theorem}[Polynomial PoA Divergence]` — Parts (i)+(ii)만 사용, Ω(N) proved
- `\begin{theorem}[Exponential PoA Divergence, conditional]` — conj:escape 가정, Ω(e^{cN})
- 헤더-본문 불일치 완전 해소
- **Cost**: 중간. Theorem 2개로 분할, 증명도 분할. 기존 reference (`Theorem thm:poa` 인용 위치 6+곳) 유지하기 위해 `thm:poa`는 polynomial 쪽에 남기고 새로운 `thm:poa_exp`를 추가.

### 경로 B: 헤더만 재명명 + 본문 유지
- 헤더 subtitle을 "(polynomial proved; exponential conditional)"로 변경
- 본문은 그대로
- **Cost**: 저. 한 줄 수정. 하지만 "theorem with two regimes" 비표준성 잔존.

### 경로 C: 현재 상태 유지 + Abstract/Intro에 cross-ref 강화
- Theorem 헤더에 `\footnotesize See Note L342 for scope` 주석
- L342 remark를 Theorem 바로 앞으로 이동
- **Cost**: 최저. 하지만 오독 위험 잔존.

## 5. 권장 경로

**경로 A (Theorem 분할)**. 이유:
1. 수학 관행 준수 (theorem 1개 = claim 1개)
2. 리뷰어 재반박 시 "우리 main result는 Theorem X (proved, unconditional), exponential은 별도 Theorem Y (conditional)"이라고 단언 가능
3. Abstract/Intro의 "Theory (proved)" 서술과 Theorem 헤더가 완전히 일치
4. Conjecture escalation이 논문 내에서 투명하게 분리됨

**비용 수용**: Theorem 2개로 분할하는 것은 Phase 2 framing 재구성의 일부로 1~2시간 작업. 증명 자체는 이미 분리되어 있으므로 증명 분할도 즉시 가능.

## 6. Phase 2 T2.1 구현 스케치

```latex
\begin{theorem}[Polynomial Price of Anarchy Divergence in TPSDs]
\label{thm:poa}
In any TPSD with $N$ agents and Byzantine fraction $\beta < 1{-}\delta^*/\phi_1^*$,
Parts~(i)--(ii) of Proposition~\ref{thm:impossibility} imply $\mathrm{PoA} \geq \Omega(N)$,
in contrast to $O(1)$ in standard social dilemmas.
\end{theorem}

\begin{proof}[Proof sketch]
[기존 "Proved Ω(N)" 블록 그대로]
\end{proof}

\begin{theorem}[Exponential Price of Anarchy Divergence, conditional on Conjecture~\ref{conj:escape}]
\label{thm:poa_exp}
Under the hypotheses of Theorem~\ref{thm:poa}, \textbf{if Conjecture~\ref{conj:escape}} holds,
then $\mathrm{PoA} \geq \Omega(e^{cN})$ for some $c>0$.
\end{theorem}

\begin{proof}[Proof sketch]
[기존 "Conjectured Ω(e^{cN})" 블록 그대로]
\end{proof}
```

## 7. Reviewer 대응

- Claude R1 `critical` (헤더 오독): Theorem 분할로 **fully addressed**.
- 재반박 시 단언: "Our main PoA result is Theorem~\ref{thm:poa} (polynomial, unconditional); the exponential version is a separate conditional theorem."

## 8. 결론

**실재하는 혼동 원인 있음**. 수학 자체는 맞지만 Theorem 헤더가 본문보다 범위가 좁아 오독 유도. Phase 2 T2.1에서 Theorem 2개로 분할하면 저비용으로 완전 해소됨. Abstract/Intro는 수정 불필요 (이미 정확함).
