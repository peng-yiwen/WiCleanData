"""Taxonomy stage configuration. See pipeline_config.py for all paths."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pipeline_config import *  # noqa: F401, F403, E402

# These two names mean the *extracted* Wikidata files in this stage,
# not the cleaned outputs used by later stages.
TAXONOMY_FILE = TAXONOMY_EXTRACTED_FILE
CLS_INST_COUNT_FILE = CLS_INST_COUNT_WIKIDATA_FILE
