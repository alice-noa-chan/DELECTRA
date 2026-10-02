#!/usr/bin/env bash
set -euo pipefail
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2
cd /root
python -c 'import torch; assert torch.__version__.startswith("2.8.0"), torch.__version__'
python -m venv --system-site-packages /root/delectra-env
source /root/delectra-env/bin/activate
python -m pip install --disable-pip-version-check --progress-bar off 'transformers==4.57.6' 'liger-kernel==0.8.4'
git clone --depth 1 --branch causal-electra https://github.com/alice-noa-chan/DELECTRA.git /root/delectra
cd /root/delectra
git fetch --depth 1 origin 82477ebe8a550146e0dd9372a758f43c90837a0c
git checkout --detach 82477ebe8a550146e0dd9372a758f43c90837a0c
test "$(git rev-parse HEAD)" = 82477ebe8a550146e0dd9372a758f43c90837a0c
mkdir -p data/tinystories-gpu-profile-256-10m-100k
echo '6251933346b8542354f061d37437073718a0d2039f80e8a7516353e914a64b52  /root/profile-data.tar.gz' | sha256sum -c -
tar -xzf /root/profile-data.tar.gz -C data/tinystories-gpu-profile-256-10m-100k
python -m pip install --disable-pip-version-check --no-deps -e .
python -m deletcra.budget_benchmark --gpu 'RTX 4090' --data-dir data/tinystories-gpu-profile-256-10m-100k --output results/runpod-4090-20261002.json --hourly-price 0.74 | tee /root/runpod-4090-20261002.log
python -m pip freeze > /root/runpod-4090-20261002-packages.txt
nvidia-smi > /root/runpod-4090-20261002-device.txt
cp results/runpod-4090-20261002.json /root/
cd /root
sha256sum runpod-4090-20261002.json runpod-4090-20261002.log runpod-4090-20261002-packages.txt runpod-4090-20261002-device.txt > runpod-4090-20261002-sha256.txt
tar -czf runpod-4090-20261002-evidence.tar.gz runpod-4090-20261002.json runpod-4090-20261002.log runpod-4090-20261002-packages.txt runpod-4090-20261002-device.txt runpod-4090-20261002-sha256.txt
sha256sum runpod-4090-20261002-evidence.tar.gz
