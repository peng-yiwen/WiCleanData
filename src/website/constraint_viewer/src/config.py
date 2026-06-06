from pathlib import Path

# Repository layout:
# constraint_viewer/
#   data/
#   assets/
#   src/
SRC_DIR = Path(__file__).resolve().parent
APP_ROOT = SRC_DIR.parent
DATA_DIR = APP_ROOT / "data"
ASSETS_DIR = APP_ROOT / "assets"

# Optional explicit model allowlist for UI/API model choices.
# Set to None (or empty list) to allow every model discovered in data/.
# Example: ["mistral7b"]
MODEL_ALLOWLIST = ["wiclean"]

# Common data files used by constraint-viewer.
REL_SUBJECT_CSV = DATA_DIR / "rel_subject_type_constraints.csv"
REL_VALUE_CSV = DATA_DIR / "rel_value_type_constraints.csv"
WIKC_LABELS = DATA_DIR / "wikclabels_2026.txt"
CLS_INST_COUNT = DATA_DIR / "cls_inst_count.txt"
# Root class https://www.wikidata.org/wiki/Q35120 (all graph nodes use ``Q`` + digits).
DEFAULT_TAXONOMY_ROOT_ID = "Q35120"
