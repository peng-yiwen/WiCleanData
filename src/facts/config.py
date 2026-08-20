"""Facts stage configuration. See pipeline_config.py for all paths."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pipeline_config import *  # noqa: F401, F403, E402


WIKIDATA_FILE = WIKIDATA_DUMP_FILE
# WIKIDATA_FILE = os.path.join(DATA_PATH, "sample.nt")   # small test dump

# In this stage TAXONOMY_FILE is the cleaned taxonomy, not the extracted dump.
# TAXONOMY_FILE = WICLEAN_TAXONOMY_FILE
VALID_CLASSES_FILE = CLASSES_FILE

# constraints to check are the cleaned constraint CSVs
TYPECHECK_SUBJ_CSV = SUBJECT_CONSTRAINTS_OUT_CSV
TYPECHECK_VALUE_CSV = VALUE_CONSTRAINTS_OUT_CSV
TYPECHECK_INST_TYPE_PATH = INST_TYPES_FILE

# constraints resimplification are the cleaned constraint CSVs
RESIMP_SUBJ_CSV = SUBJECT_CONSTRAINTS_OUT_CSV
RESIMP_VALUE_CSV = VALUE_CONSTRAINTS_OUT_CSV
RESIMP_INST_TYPE_PATH = INST_TYPES_FILE
