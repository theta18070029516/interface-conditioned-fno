#!/usr/bin/env bash
set -euo pipefail

PHYSICAL_GPU_INDEX="${1:-}"
if [[ -z "${PHYSICAL_GPU_INDEX}" ]]; then
  echo "用法: bash deep_fno_shared_benchmark/remote_validate_v2.sh <物理GPU编号>" >&2
  exit 2
fi
if [[ "${PHYSICAL_GPU_INDEX}" == "6" ]]; then
  echo "拒绝运行：物理 GPU 6 已永久排除。" >&2
  exit 2
fi

PROJECT_PARENT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_PARENT}"

CUDA_VISIBLE_DEVICES="${PHYSICAL_GPU_INDEX}" \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
python -m unittest discover -s deep_fno_shared_benchmark/tests -v
python -m deep_fno_shared_benchmark.gpu_smoke_v2 \
  --physical-gpu "${PHYSICAL_GPU_INDEX}"

echo "PASS: 完整测试与 v2 GPU smoke test 均通过。"
