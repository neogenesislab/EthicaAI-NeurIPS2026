---
description: EthicaAI 개발 환경 셋업 및 실행
---

# EthicaAI 개발 워크플로우

## 1단계: 스킬 설치 (새 IDE에서 1회)
```powershell
npx -y antigravity-awesome-skills --path "D:\00.test\PAPER\EthicaAI_anon2\.agent\skills"
```

## 2단계: Python 환경 (1회, WhyLab과 공유 가능)
```powershell
conda activate whylab
```

```powershell
pip install pettingzoo[classic] stable-baselines3 gymnasium supersuit tensorboard
```

## 3단계: 시뮬레이션 실행
// turbo
```powershell
conda activate whylab && python D:\00.test\PAPER\EthicaAI_anon2\simulation\run_experiment.py
```

## 4단계: Jupyter 분석
```powershell
conda activate whylab && cd D:\00.test\PAPER\EthicaAI_anon2\analysis && jupyter lab
```

## 5단계: 텐서보드 모니터링
// turbo
```powershell
conda activate whylab && tensorboard --logdir D:\00.test\PAPER\EthicaAI_anon2\simulation\logs
```
