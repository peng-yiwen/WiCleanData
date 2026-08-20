#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
export PYTHONPATH="../facts:${PYTHONPATH:-}"

# Read LLM model list from config.py
mapfile -t LLMs < <(python -c "import config; print('\n'.join(config.LLM_MODELS))")

echo "=== Step 1: Extract taxonomy from Wikidata dump ==="
python extract.py # CPUs

echo "=== Step 2: LLM inference on taxonomy edges ===" # GPUs
for llm in "${LLMs[@]}"; do
    echo "  Running inference with model: $llm"
    python llm_infer.py --llm "$llm"
done

echo "=== Step 3: Rewire link inference ==="
python llm_rewire.py

echo "=== Step 4: Taxonomy refinement (cut/resolve/reduce/merge/filter) ==="
python main.py # CPUs

echo "=== All steps completed ==="
