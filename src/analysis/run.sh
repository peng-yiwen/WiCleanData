#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

log() { echo; echo "========================================"; echo "  $*"; echo "========================================"; }

log "Step 1/4 — Dataset Statistics (stats.py)"
python stats.py

log "Step 2/4 — Taxonomy Robustness (robust.py)"
python robust.py

# log "Step 3/4 — Concept Similarity Correlation (csc.py)"
# python csc.py

# log "Step 4/4 — Graph Edit Distance (ged.py)"
# python ged.py

log "All analysis steps completed."
