"""
Centralized path configuration for the constraints pipeline.
Edit paths here to adapt to different environments or data versions.
"""

from pathlib import Path
import os


CONSTRAINTS_DIR = Path(__file__).resolve().parent
DATA_DIR = '../../data/'

# taxonomy and mapping
ROOT_QID = "Q35120"
TAXONOMY_PATH = os.path.join(DATA_DIR, 'wicleanData', 'wicleanTaxonomy.txt')
MAPPING_PATH = os.path.join(DATA_DIR, 'wicleanData', 'wiclean_mapping.txt')
TAXONOMY_LABELS_FILE = os.path.join(DATA_DIR, 'wikidata', 'wiki_taxonomy_extracted_labels.tsv')

# input constraint csvs
SUBJECT_CONSTRAINTS_CSV = os.path.join(DATA_DIR, 'wikidata', 'subject_type_constraints.csv')
VALUE_CONSTRAINTS_CSV = os.path.join(DATA_DIR, 'wikidata', 'value_type_constraints.csv')

# output constraint csvs
SUBJECT_CONSTRAINTS_OUT_CSV = os.path.join(DATA_DIR, 'wicleanData', 'subject_type_constraints_clean.csv')
VALUE_CONSTRAINTS_OUT_CSV = os.path.join(DATA_DIR, 'wicleanData', 'value_type_constraints_clean.csv')

# instance counts (used by metrics for IC computation)
WICLEAN_OUTPUT_DIR = os.path.join(DATA_DIR, 'wicleanData')
CLS_INSTANCE_COUNT_PATH = os.path.join(WICLEAN_OUTPUT_DIR, "cls_inst_count.csv")
# os.path.join(DATA_DIR, 'wikidata', 'class_instance_count.csv')

# thresholds
THRESHOLDS = {
    "avg_distance": 2,
    "ic_difference": -0.1,
    # "expansion_ratio": 0.5,
    # "depth_ratio": 0.5,
}