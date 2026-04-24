# =============================================================
# EthicaAI WSL2 GPU 원스텝 자동 셋업
# 재부팅 후 관리자 PowerShell에서 이 스크립트 실행하세요:
#   powershell -ExecutionPolicy Bypass -File D:\00.test\PAPER\EthicaAI_anon2\scripts\auto_gpu_setup.ps1
# =============================================================

Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "EthicaAI GPU Auto Setup (Post-Reboot)" -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan

# Step 1: Ubuntu 설치
Write-Host "`n[1/4] Installing Ubuntu 24.04..." -ForegroundColor Yellow
wsl --install --distribution Ubuntu-24.04 --no-launch 2>&1
wsl --set-default-version 2 2>&1

Write-Host "`n[2/4] Launching Ubuntu for first-time setup..." -ForegroundColor Yellow
Write-Host "  -> Ubuntu 창이 열리면 사용자명/비밀번호를 설정하세요" -ForegroundColor White
Write-Host "  -> 설정 후 'exit' 입력하면 이 스크립트가 계속됩니다" -ForegroundColor White
wsl --distribution Ubuntu-24.04

# Step 3: JAX GPU 환경 구축
Write-Host "`n[3/4] Setting up JAX GPU environment..." -ForegroundColor Yellow
wsl --distribution Ubuntu-24.04 -- bash -c "
set -e
echo '=== System Update ==='
sudo apt-get update -qq && sudo apt-get install -y -qq python3 python3-pip python3-venv

echo '=== Creating Virtual Environment ==='
python3 -m venv ~/ethica_env
source ~/ethica_env/bin/activate

echo '=== Installing JAX with CUDA 12 ==='
pip install --upgrade pip
pip install 'jax[cuda12]' scipy matplotlib

echo '=== GPU Verification ==='
python3 -c \"
import jax
print('JAX version:', jax.__version__)
print('Devices:', jax.devices())
print('Backend:', jax.default_backend())
gpu_ok = jax.default_backend() != 'cpu'
print('GPU READY:', gpu_ok)
if gpu_ok:
print('[SUCCESS] RTX 4070 SUPER detected!')
else:
print('[WARNING] GPU not detected, running on CPU')
\"
echo '=== Environment Setup Complete ==='
"

# Step 4: 실험 실행 안내
Write-Host "`n=========================================" -ForegroundColor Green
Write-Host "Setup Complete! GPU 환경이 준비되었습니다." -ForegroundColor Green
Write-Host "=========================================" -ForegroundColor Green
Write-Host "`n실험 실행 방법:" -ForegroundColor White
Write-Host "  wsl --distribution Ubuntu-24.04" -ForegroundColor White
Write-Host "  source ~/ethica_env/bin/activate" -ForegroundColor White
Write-Host "  cd /mnt/d/00.test/PAPER/EthicaAI_anon2" -ForegroundColor White
Write-Host "  python simulation/jax/run_full_pipeline.py baseline" -ForegroundColor White
Write-Host ""
pause
