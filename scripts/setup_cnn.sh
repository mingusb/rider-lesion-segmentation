#!/usr/bin/env bash
set -euo pipefail
root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
external="$root/external"
mkdir -p "$external"
if [ ! -d "$external/LesionLocator/.git" ]; then
  git clone https://github.com/MIC-DKFZ/LesionLocator.git "$external/LesionLocator"
fi
if [ -n "$(git -C "$external/LesionLocator" status --porcelain)" ]; then echo 'External checkout has changes; preserve it and use a fresh directory.' >&2; exit 1; fi
git -C "$external/LesionLocator" checkout --detach ef073baaa0dcba8dd036ec5a29f130db18c9d580
uv venv --python 3.12.14 "$root/.venv"
uv pip install --python "$root/.venv/bin/python" --extra-index-url https://download.pytorch.org/whl/cu132 --index-strategy unsafe-best-match -r "$root/scripts/lesionlocator/requirements.lock.txt"
uv pip install --python "$root/.venv/bin/python" --no-deps -e "$external/LesionLocator"
uv pip install --python "$root/.venv/bin/python" xgboost==3.4.1
"$root/.venv/bin/python" -c 'import torch, lesionlocator, xgboost; print("torch",torch.__version__,"xgboost",xgboost.__version__)'
