#!/bin/bash
set -e

echo "==========================================="
echo "Melting Pot Environment Setup (WSL / Ubuntu)"
echo "==========================================="

echo "[1/4] Installing Miniconda..."
wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
bash /tmp/miniconda.sh -b -u -p ~/miniconda3
rm /tmp/miniconda.sh

echo "[2/4] Initializing conda..."
source ~/miniconda3/bin/activate

echo "[3/4] Creating Python 3.11 environment 'mp_env'..."
# Accept Conda terms of service
~/miniconda3/bin/conda config --set auto_activate_base false || true
~/miniconda3/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true
~/miniconda3/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true

# dm-meltingpot 2.3.1+ requires Python >= 3.11
~/miniconda3/bin/conda create -y -n mp_env python=3.11

echo "[4/4] Activating mp_env and installing dependencies..."
conda activate mp_env
# dm-meltingpot 2.2.0 or 2.3.1 usually works best on Linux wheels
pip install dm-meltingpot==2.3.1
pip install ray[rllib]==2.9.3 pettingzoo==1.24.3 supersuit==3.9.2 numpy==1.26.4

echo "==========================================="
echo "SUCCESS! Melting Pot is installed."
echo "To use it, run: source ~/miniconda3/bin/activate mp_env"
echo "==========================================="
