#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
export PYTHONPATH="../facts:${PYTHONPATH:-}"

python extract.py
python llm_infer.py --llm gemma27b
python refine.py
