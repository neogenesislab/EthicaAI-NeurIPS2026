# NeurIPS 2026 Paper Checklist (사전 작성)

> NeurIPS는 모든 제출 논문에 이 체크리스트를 의무적으로 포함해야 합니다.
> 아래는 EthicaAI 논문에 맞게 미리 답변한 버전입니다.

## 1. Claims
- [x] 논문의 주요 주장(claims)이 Abstract과 Introduction에 명확히 기술되어 있는가?
  - Yes. C1-C4 4개의 Contribution이 명시적으로 나열됨.

## 2. Limitations
- [x] 논문이 limitations를 논의하는가?
  - Yes. Section 5.3 (Limitations)에서 에이전트 수, 환경 종류, 인간 실험 부재 등을 논의.

## 3. Theory
- [ ] N/A — 본 논문은 이론 증명 논문이 아닌 실험 논문.

## 4. Experiments
- [x] 모든 실험에 대해 training/evaluation 세부 정보가 충분한가?
  - Yes. Appendix A에 전체 하이퍼파라미터 테이블 포함.
- [x] 에러 바 또는 신뢰구간이 보고되는가?
  - Yes. OLS HAC Robust SE, LMM, Bootstrap 95% CI 모두 보고.
- [x] 여러 seeds로 실험이 반복되는가?
  - Yes. 10개 seeds (0, 42, 123, 256, 999, 1337, 2024, 3141, 4269, 5555).
- [x] 통계적 유의성 검정이 적절히 수행되는가?
  - Yes. 3중 검증: OLS(HAC) + LMM + Bootstrap CI.

## 5. Reproducibility
- [x] 알고리즘, 데이터 전처리, 실험 설정의 충분한 세부 정보가 제공되는가?
  - Yes. 전체 코드가 GitHub에 공개, requirements.txt 포함.
- [x] 코드가 제출되거나 공개 약속이 되어 있는가?
  - Yes. https://anonymous.4open.science/r/EthicaAI (MIT License).
- [x] 실험 결과 재현에 필요한 자원(compute)이 명시되어 있는가?
  - Yes. RTX 4070 SUPER (12GB VRAM), ~35분/sweep.

## 6. Broader Impact
- [x] 잠재적 사회적 영향이 논의되는가?
  - Yes. AI Alignment 시사점을 Discussion에서 논의. 상황적 헌신(Situational Commitment)의 
    윤리적 함의와 한계.

## 7. Safeguards
- [x] 악용 가능성에 대해 논의하는가?
  - Partial. 메타랭킹이 조작에 사용될 수 있는 가능성을 Limitations에서 언급.

## 8. Licenses
- [x] 사용된 자산/데이터의 라이선스가 명시되는가?
  - Yes. MIT License. 외부 데이터 사용 없음.

## 9. New Assets
- [x] 새로운 데이터셋/코드가 적절히 문서화되는가?
  - Yes. GitHub README + Zenodo DOI.

## 10. Human Subjects
- [ ] N/A — 인간 피험자 실험 없음.
