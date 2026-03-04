---
description: NeurIPS 제출물 종합 감사 (audit) 실행
---

# NeurIPS 제출물 종합 감사 워크플로우

이 워크플로우는 제출물의 **모든 문서를 글자 단위로 자동 검증**합니다.

## 사전 조건

- `NeurIPS2026_final_submission/` 디렉터리가 존재할 것
- Python 3.8+ 설치

## 실행 단계

// turbo-all

### 1. 감사 스크립트 실행

```powershell
python d:\00.test\PAPER\EthicaAI\NeurIPS2026_final_submission\code\scripts\audit_submission.py
```

이 스크립트는 8개 모듈로 구성됩니다:

| 모듈 | 검증 항목 |
|---|---|
| Module 1 | TeX ↔ JSON 수치 교차 검증 |
| Module 2 | BibTeX 무결성 (미참조/미정의 엔트리) |
| Module 3 | 그림 파일 존재 (`\includegraphics` 전수 검사) |
| Module 4 | 플레이스홀더 잔류 (X%, TODO, FIXME, TBD, ???) |
| Module 5 | LaTeX 빌드 로그 분석 (에러/경고/Overfull) |
| Module 6 | README 팩트체크 (경로/파일명 실존 확인) |
| Module 7 | 코드 일관성 (출력 경로, import, requirements pinning) |
| Module 8 | TeX 교차참조 (\\ref ↔ \\label 대응) |

### 2. 결과 확인

감사 보고서는 `NeurIPS2026_final_submission/audit_report.txt`에 UTF-8로 저장됩니다.

- **FAIL**: 반드시 수정해야 하는 치명적 오류
- **WARN**: 검토가 필요하지만 제출 가능한 경고
- **INFO**: 참고 정보

### 3. FAIL 항목 수정 후 재실행

수정 후 감사 스크립트를 다시 실행하여 **FAIL 0건 (PASS ✅)** 을 확인합니다.

### 4. PDF 재빌드

```powershell
cd d:\00.test\PAPER\EthicaAI\NeurIPS2026_final_submission\paper
pdflatex -interaction=nonstopmode unified_paper.tex; bibtex unified_paper; pdflatex -interaction=nonstopmode unified_paper.tex; pdflatex -interaction=nonstopmode unified_paper.tex
```

### 5. ZIP 패키징

```powershell
python d:\00.test\PAPER\EthicaAI\NeurIPS2026_final_submission\code\scripts\build_submission_zip.py
```

### 6. Git 커밋

```powershell
cd d:\00.test\PAPER\EthicaAI
git add -A; git commit -m "audit: fix all findings from audit_submission.py"; git push
```
