#!/usr/bin/env bash
# Pipeline runner — executes all steps in order.
# Run from the src/ directory: bash run.sh

set -euo pipefail

SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SRC_DIR"

log() { echo; echo "========================================"; echo "  $*"; echo "========================================"; }

log "Step 1/5 — ParseInstanceTypes"
python ParseInstanceTypes.py

log "Step 2/5 — GetNoLabeledInstance"
python GetNoLabeledInstance.py

log "Step 3/5 — ParseWikiFacts"
python ParseWikiFacts.py

log "Step 4/5 — FactsTypeCheck"
python FactsTypeCheck.py

log "Step 5/5 — ConstraintResimplification"
python ConstraintResimplification.py

log "All steps completed successfully."
