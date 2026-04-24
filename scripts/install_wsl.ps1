# =============================================================
# EthicaAI WSL2 GPU 환경 구축 (관리자 권한 필요)
# PowerShell을 관리자로 열고 이 스크립트를 실행하세요:
# powershell -ExecutionPolicy Bypass -File scripts\install_wsl.ps1
# =============================================================

Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "EthicaAI WSL2 GPU Setup - Step 1" -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan

# WSL2 + Ubuntu 설치
Write-Host "[1/2] WSL2 + Ubuntu 24.04 설치 중..." -ForegroundColor Yellow
wsl --install --distribution Ubuntu-24.04

Write-Host ""
Write-Host "[2/2] 설치 완료 후 재부팅이 필요할 수 있습니다." -ForegroundColor Yellow
Write-Host ""
Write-Host "=========================================" -ForegroundColor Green
Write-Host "재부팅 후 다음 단계:" -ForegroundColor Green
Write-Host "  1. Ubuntu 터미널을 열고 사용자 계정을 생성하세요" -ForegroundColor White
Write-Host "  2. Ubuntu 터미널에서 다음 명령어를 실행하세요:" -ForegroundColor White
Write-Host "     cd /mnt/d/00.test/PAPER/EthicaAI_anon2" -ForegroundColor White
Write-Host "     bash scripts/wsl_setup.sh" -ForegroundColor White
Write-Host "=========================================" -ForegroundColor Green
Write-Host ""
Write-Host "설치 완료 후 GPU 실험 실행:" -ForegroundColor White
Write-Host "  source ~/ethica_env/bin/activate" -ForegroundColor White
Write-Host "  cd /mnt/d/00.test/PAPER/EthicaAI_anon2" -ForegroundColor White
Write-Host "  python simulation/jax/run_full_pipeline.py baseline" -ForegroundColor White
Write-Host ""
pause
