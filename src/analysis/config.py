"""Analysis stage configuration. See pipeline_config.py for all paths."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pipeline_config import *  

# In this stage these names refer to cleaned pipeline outputs,
# not the extracted dump / intermediate facts filename.
TAXONOMY_FILE = WICLEAN_TAXONOMY_FILE
LABELS_FILE = TAXONOMY_LABELS_FILE
FACTS_FILE = WICLEAN_FACTS_FILE
# the cleaned constraint CSVs
SUBJ_CONSTRAINTS_FILE = SUBJECT_CONSTRAINTS_OUT_CSV
VALUE_CONSTRAINTS_FILE = VALUE_CONSTRAINTS_OUT_CSV