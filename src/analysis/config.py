"""
Centralized path configuration for the analysis pipeline.
Edit paths here to adapt to different environments or data versions.
"""

import os

# ===========================================================================
#  Base directories (relative to this file: src/analysis/)
# ===========================================================================

DATA_DIR = os.path.join(os.path.dirname(__file__), '../../data/')
WIKIDATA_DIR = os.path.join(DATA_DIR, 'wikidata')
WICLEAN_DIR = os.path.join(DATA_DIR, 'wicleanData')
ROOT_QID = 'Q35120'

# ===========================================================================
#  Taxonomy & labels (bare QIDs, no 'wd:' prefix)
# ===========================================================================

TAXONOMY_FILE = os.path.join(WICLEAN_DIR, 'wicleanTaxonomy.txt')
LABELS_FILE = os.path.join(WIKIDATA_DIR, 'wiki_2026_extracted_labels.tsv')
EMBEDDING_PKL_FILE = os.path.join(WIKIDATA_DIR, 'wiki_2026_labels_emb.pkl')

# ===========================================================================
#  Facts & instance types 
# ===========================================================================

FACTS_FILE = os.path.join(WICLEAN_DIR, 'wiclean_facts.tsv')
INST_TYPES_FILE = os.path.join(WICLEAN_DIR, 'instTypes/', 'wiki_instance_types.tsv')
CLS_INST_COUNT_FILE = os.path.join(WICLEAN_DIR, 'cls_inst_count.csv')

# ===========================================================================
#  Constraints  
# ===========================================================================

SUBJ_CONSTRAINTS_FILE = os.path.join(WICLEAN_DIR, 'subject_type_constraints_clean.csv')
VALUE_CONSTRAINTS_FILE = os.path.join(WICLEAN_DIR, 'value_type_constraints_clean.csv')

# ===========================================================================
#  Output directories
# ===========================================================================

STATS_OUTPUT_DIR = os.path.join(WICLEAN_DIR, 'statistics')
ROBUSTNESS_OUTPUT_FILE = os.path.join(WICLEAN_DIR, 'analysis/', 'robustness.txt')
