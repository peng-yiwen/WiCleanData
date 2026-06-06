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

# Path to the raw Wikidata NT dump
WIKIDATA_FILE = os.path.join("/projects/dig", "latest-truthy.nt") # the whole truthy dump of Wikidata
# WIKIDATA_FILE = os.path.join(DATA_PATH, "sample.nt")   # small test dump

# Shared reference files (under DATA_PATH)
TAXONOMY_FILE          = os.path.join(DATA_PATH, "wicleanTaxonomy.txt")
TAXONOMY_BEFORE_WP_FILE = os.path.join(DATA_PATH, "wicleanTaxonomy_before_wikipedia_filtering.txt")
WICLEAN_MAPPING_FILE   = os.path.join(DATA_PATH, "wiclean_mapping.txt")

# ===========================================================================
#  ParseInstanceTypes.py
# ===========================================================================

INST_TYPES_FOLDER      = os.path.join(DATA_PATH, "instTypes/")
INST_META_MESSAGES_FILE = "wiki_instance_types_meta_messages.log"
INST_FACTS_FILE        = "wiki_instance_types.tsv"

# Input reference files
VALID_CLASSES_FILE     = os.path.join(DATA_PATH, "valid_classes_after_extract.txt")

# ===========================================================================
#  GetNoLabeledInstance.py
# ===========================================================================

# Input : INST_TYPES_FOLDER + INST_META_MESSAGES_FILE  (defined above)
# Output: NO_LABEL_INSTANCES_FILE                       (defined below, shared with ParseWikiFacts.py)


# ===========================================================================
#  ParseWikiFacts.py
# ===========================================================================

FACTS_FOLDER           = os.path.join(DATA_PATH, "facts/")
FACTS_META_MESSAGES_FILE = "wiki_facts_meta_messages.log"
FACTS_FILE             = "wiki_facts.tsv"

# Input reference files
IDENTIFIERS_FILE       = os.path.join(DATA_PATH, "identifiers.txt")
NO_LABEL_INSTANCES_FILE = os.path.join(DATA_PATH, "instTypes/no_label_instances.txt")
SCHOLARLY_ARTICLE_CLASSES_FILE = os.path.join(DATA_PATH, "ScholarlyArticleClasses.txt")

# ===========================================================================
#  FactsTypeCheck.py
# ===========================================================================

TYPECHECK_SUBJ_CSV  = os.path.join(DATA_PATH, "rel_constraints_subj_types_clean.csv")
TYPECHECK_VALUE_CSV = os.path.join(DATA_PATH, "rel_constraints_value_types_clean.csv")

# Input: instance types produced by ParseInstanceTypes.py
TYPECHECK_INST_TYPE_PATH = os.path.join(INST_TYPES_FOLDER, INST_FACTS_FILE)
TYPECHECK_FACTS_PATH = os.path.join(FACTS_FOLDER, FACTS_FILE)
# TYPECHECK_FACTS_PATH = os.path.join(FACTS_FOLDER, "wiki_facts_no_identifier_no_article.tsv")

TYPECHECK_OUTPUT_FOLDER      = FACTS_FOLDER
TYPECHECK_META_MESSAGES_FILE = "wiki_facts_constrainted_meta_messages.log"
TYPECHECK_FACTS_FILE         = "wiki_facts_constrainted.tsv"

# ===========================================================================
#  ConstraintResimplification.py
# ===========================================================================

RESIMP_SUBJ_CSV   = os.path.join(DATA_PATH, "rel_constraints_subj_types_clean.csv")
RESIMP_VALUE_CSV  = os.path.join(DATA_PATH, "rel_constraints_value_types_clean.csv")
RESIMP_INST_TYPE_PATH = os.path.join(INST_TYPES_FOLDER, INST_FACTS_FILE)
RESIMP_FACTS_PATH = os.path.join(FACTS_FOLDER, TYPECHECK_FACTS_FILE)
RESIMP_UNUSED_TYPES_CSV = os.path.join(DATA_PATH, "constraints/rel_constraints_unused_types.csv")
