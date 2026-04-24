# EthicaAI 워크스페이스 룰

## 프로젝트 정체성
- **프로젝트명**: EthicaAI — Amartya Sen의 Meta-Ranking 이론 계산적 검증
- **한 줄**: MARL 시뮬레이션으로 "상황적 도덕 에이전트가 진화적으로 안정함"을 증명
- **목표**: NeurIPS 2026 Main Track 투고 (+ ICSD 6월, AIES 10월)
- **형태**: 학술 논문 + 시뮬레이션 코드 + 재현 파이프라인 + 시각화 대시보드

---

## 우선순위 정책 (Priority Policy)

### P0: 즉시 실행 (묻지 않고 실행)
- reproduce.py 모듈 실행
- Figure 생성/복사
- Git commit + push
- 코드 정적 분석/린트

### P1: 빠르게 확인 후 실행
- 새 분석 스크립트 작성
- 논문/LaTeX 수정
- requirements.txt / Dockerfile 수정
- README 업데이트

### P2: 사용자 확인 필요
- 외부 패키지 대대적 업그레이드 (jax 버전 변경 등)
- 데이터 삭제/구조 변경
- 새로운 Phase/실험 방향 결정

### P3: 반드시 사전 승인
- API 키/인증 관련 작업
- 원격 서버 배포
- public 저장소 민감 정보 노출 위험

---

## 컴퓨팅 환경 정책

### 현재 환경
- **GPU**: RTX 4070 SUPER (12GB VRAM)
- **Windows**: JAX/PyTorch CPU only (CUDA 미지원)
- **WSL2 Ubuntu 24.04**: ✅ **JAX GPU 활성화** (CUDA, venv ~/ethicaai_env)
- **Python**: Windows 3.13(miniconda) / WSL 3.12(venv)

### GPU 실행 정책 (필수)
1. **시뮬레이션/분석 실행**: 반드시 WSL2 GPU로 실행
   ```bash
   wsl -d Ubuntu-24.04 -- bash -c "source ~/ethicaai_env/bin/activate && cd /mnt/d/00.test/PAPER/EthicaAI_anon2 && python3 reproduce.py"
   ```
2. **빠른 단일 모듈**: WSL2 경유
   ```bash
   wsl -d Ubuntu-24.04 -- bash -c "source ~/ethicaai_env/bin/activate && cd /mnt/d/00.test/PAPER/EthicaAI_anon2 && python3 -m simulation.jax.analysis.<모듈> simulation/outputs/reproduce"
   ```
3. **파일 편집/Git/문서**: Windows에서 직접 (GPU 불필요)
4. **벤치마크**: 5000×5000 matmul = **12.8ms** (GPU) vs ~2000ms (CPU) → **~150배 빠름**

### 자동 실행 정책
- `python -m simulation.*` → SafeToAutoRun=true
- `git status/log/diff` → SafeToAutoRun=true
- `git add/commit/push` → SafeToAutoRun=false (사용자 확인)
- 파이프라인(`|`) 명령 → 단일 표현식으로 변환 (Allow List 호환)

---

## 코드 규칙

### 언어
- 모든 코드 주석, 커밋, 문서: **한국어**
- 수학 용어/변수명: 영문 유지 (λ_t, ATE, SVO, ESS)
- 논문 본문: 한국어(`paper_korean.md`) + 영문(`paper_english.md`)
- LaTeX: 영문 (`paper/neurips2026_main.tex`)

### Python 코드
- Python 3.10+
- **JAX 기반** 시뮬레이션 (jax, flax, optax, distrax)
- 모든 함수에 타입 힌트 필수
- 독스트링: Google 스타일
- 출력 인코딩: `PYTHONIOENCODING=utf-8` 필수

### 프로젝트 구조
```
EthicaAI/
├── simulation/jax/analysis/   # 38 분석 모듈 (reproduce.py에 등록)
├── simulation/jax/environments/  # 환경 (Cleanup, IPD, PGG 등)
├── simulation/outputs/reproduce/ # Figure + JSON 결과
├── paper/                     # NeurIPS LaTeX (Main + Supp + Bib)
├── site/figures/              # 대시보드용 Figure 복사본
├── reproduce.py               # 전체 재현 (38모듈)
├── prepare_arxiv.py           # arXiv 패키지 생성
├── Dockerfile                 # Docker 재현성
└── requirements.txt           # Python 의존성
```

### 시뮬레이션 설계 원칙
- 에이전트 SVO 7종: selfish(0°), individualist(15°), competitive(-15°),
  cooperative(60°), prosocial(45°), altruistic(90°), competitive(-30°)
- 환경 8종: Cleanup, IPD, PGG, Harvest, Climate, Vaccine, Governance, Network
- 모든 실험: **시드 고정**(10 seeds) + 재현 가능
- 새 모듈 추가 시: reproduce.py ANALYSES에 반드시 등록

### 수학적 엄밀함
- Meta-Ranking: R_total = (1-λ_t)·U_self + λ_t·[U_meta - ψ]
- 통계: LMM + Cluster Bootstrap SE + Causal Forest HTE
- 모든 결과에 p-value, 효과 크기(Cohen's f²), 신뢰구간 포함

---

## Git
- 커밋: `Phase X: 한국어 설명 (Fig N~M, 주요 결과)`
- 브랜치: main 단일 (연구 개발 프로젝트)
- push 전 보안 체크: API 키, 개인정보 노출 없음 확인

## 보안
- API 키 커밋 금지
- 합성/시뮬레이션 데이터만 사용
- 개인 이메일 외 민감정보 없음

## 현재 상태 (2026-02-15)
- **Phase A~R**: 전체 완료 (74 Figure, 38 모듈, 35 섹션)
- **Q1 인간 실험**: IRB + 실험실 필요 (미완)
- **다음 단계**: arXiv 업로드 → NeurIPS 제출 → ICSD/AIES 확장
