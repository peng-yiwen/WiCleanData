"""
Central configuration file for all path and filename settings.
Edit this file to change any input/output paths across the pipeline.
"""

import os

# ===========================================================================
#  Shared base paths
# ===========================================================================

# Root data directory (relative to the src/ folder)
DATA_PATH = "../../data/"
WIKIDATA_DIR = os.path.join(DATA_PATH, "wikidata")
WICLEAN_DIR = os.path.join(DATA_PATH, "wicleanData")

# Path to the raw Wikidata NT dump
WIKIDATA_FILE = os.path.join(WIKIDATA_DIR, "latest-truthy.nt") # the whole truthy dump of Wikidata
# WIKIDATA_FILE = os.path.join(DATA_PATH, "sample.nt")   # small test dump

TAXONOMY_FILE          = os.path.join(WICLEAN_DIR, "wicleanTaxonomy.txt")
TAXONOMY_BEFORE_WP_FILE = os.path.join(WICLEAN_DIR, "wicleanTaxonomy_before_wikipedia_filtering.txt")
WICLEAN_MAPPING_FILE   = os.path.join(WICLEAN_DIR, "wiclean_mapping.txt")

# ===========================================================================
#  ParseInstanceTypes.py
# ===========================================================================

INST_TYPES_FOLDER      = os.path.join(WICLEAN_DIR, "instTypes/")
INST_META_MESSAGES_FILE = "wiki_instance_types_meta_messages.log"
INST_FACTS_FILE        = "wiki_instance_types.tsv"

# Input reference files
VALID_CLASSES_FILE     = os.path.join(WIKIDATA_DIR, "classes.csv")


# ===========================================================================
#  ParseWikiFacts.py
# ===========================================================================

FACTS_FOLDER           = os.path.join(WICLEAN_DIR, "facts/")
FACTS_META_MESSAGES_FILE = "wiki_facts_meta_messages.log"
FACTS_FILE             = "wiki_facts.tsv"

# Input reference files
IDENTIFIERS_FILE       = os.path.join(DATA_PATH, "external_identifier.csv")
NO_LABEL_INSTANCES_FILE = os.path.join(WICLEAN_DIR, "instTypes/no_label_instances.txt")
SCHOLARLY_ARTICLE_CLASSES_FILE = os.path.join(DATA_PATH, "scholarly_articles.csv")

# ===========================================================================
#  FactsTypeCheck.py
# ===========================================================================

TYPECHECK_SUBJ_CSV  = os.path.join(WICLEAN_DIR, "subject_constraints_types_clean.csv")
TYPECHECK_VALUE_CSV = os.path.join(WICLEAN_DIR, "value_constraints_types_clean.csv")

# Input: instance types produced by ParseInstanceTypes.py
TYPECHECK_INST_TYPE_PATH = os.path.join(INST_TYPES_FOLDER, INST_FACTS_FILE)
TYPECHECK_FACTS_PATH = os.path.join(FACTS_FOLDER, FACTS_FILE)

TYPECHECK_OUTPUT_FOLDER      = WICLEAN_DIR
TYPECHECK_META_MESSAGES_FILE = "wiclean_facts_constrainted_meta_messages.log"
TYPECHECK_FACTS_FILE         = "wiclean_facts.tsv"

# ===========================================================================
#  ConstraintResimplification.py
# ===========================================================================

RESIMP_SUBJ_CSV   = os.path.join(WICLEAN_DIR, "subject_constraints_types_clean.csv")
RESIMP_VALUE_CSV  = os.path.join(WICLEAN_DIR, "value_constraints_types_clean.csv")
RESIMP_INST_TYPE_PATH = os.path.join(INST_TYPES_FOLDER, INST_FACTS_FILE)
RESIMP_FACTS_PATH = os.path.join(WICLEAN_DIR, TYPECHECK_FACTS_FILE)
RESIMP_UNUSED_TYPES_CSV = os.path.join(WICLEAN_DIR, "constraints/rel_constraints_unused_types.csv")

# Creat the folder if it doesn't exist
for d in [INST_TYPES_FOLDER, FACTS_FOLDER, 
          os.path.dirname(RESIMP_UNUSED_TYPES_CSV)]:
    os.makedirs(d, exist_ok=True)