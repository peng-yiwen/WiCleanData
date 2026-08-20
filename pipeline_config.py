"""
Unified configuration for the WiCleanData pipeline.
All path definitions live here. Each stage's config.py imports this file
and only overrides names that mean different things in that stage.

To override the data directory:
    export WICLEAN_DATA_DIR=/path/to/your/data
"""

import os
from pathlib import Path

# ===========================================================================
#  Project root & base directories
# ===========================================================================

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("WICLEAN_DATA_DIR", PROJECT_ROOT / "data"))
WIKIDATA_DIR = DATA_DIR / "wikidata"
WICLEAN_DIR = DATA_DIR / "wicleanData"
WIKIPEDIA_DIR = WIKIDATA_DIR / "wikipedia"
ROOT_QID = "Q35120"


# ===========================================================================
#  Raw Wikidata inputs (from SPARQL / dump)
# ===========================================================================

WIKIDATA_DUMP_FILE = WIKIDATA_DIR / "latest-truthy.nt"
CLASSES_FILE = WIKIDATA_DIR / "classes.csv"
METACLASSES_FILE = WIKIDATA_DIR / "metaclasses.csv"
BFO_CLASSES_FILE = WIKIDATA_DIR / "bfo_classes.csv"
CLS_INST_COUNT_WIKIDATA_FILE = WIKIDATA_DIR / "class_instance_count.csv"

# ===========================================================================
#  Taxonomy stage
# ===========================================================================

TAXONOMY_EXTRACTED_FILE = WIKIDATA_DIR / "wiki_taxonomy_extracted.tsv"
TAXONOMY_LABELS_FILE = WIKIDATA_DIR / "wiki_taxonomy_extracted_labels.tsv"
TAXONOMY_DESCRIPTIONS_FILE = WIKIDATA_DIR / "wiki_taxonomy_extracted_descriptions.tsv"
EMBEDDING_PKL_FILE = WIKIDATA_DIR / "wiki_2026_labels_emb.pkl"
# LABELS_FILE = WIKIDATA_DIR / "wiki_2026_extracted_labels.tsv"

RESULTS_DIR = PROJECT_ROOT / "results"
LLM_OUTPUT_DIR = RESULTS_DIR / "llm_output"
INTERMEDIATE_GRAPHS_DIR = RESULTS_DIR / "intermediate_graphs"

PROMPTS_DIR = PROJECT_ROOT / "prompts"
SUBCLASS_EVAL_PROMPT = PROMPTS_DIR / "SubClassEval.txt"

# Final taxonomy outputs
WICLEAN_TAXONOMY_FILE = WICLEAN_DIR / "wicleanTaxonomy.txt"
WICLEAN_TAXONOMY_BEFORE_WP_FILE = WICLEAN_DIR / "wicleanTaxonomy_before_wikipedia_filtering.txt"
WICLEAN_MAPPING_FILE = WICLEAN_DIR / "wiclean_mapping.txt"
WICLEAN_LABELS_FILE = WICLEAN_DIR / "wicleanLabels.txt"

# LLM settings
LLM_MODELS = ["mistral24b", "gemma27b", "qwen32b"]
MAJORITY_PREDICTIONS_FILE = "llm_majority_predictions.txt"
MAJORITY_REWIRE_LINKS_FILE = "majority_rewire_links.txt"
MAJORITY_PREDICTIONS_REWIRE_FILE = "majority_predictions_rewire.json"

# ===========================================================================
#  Constraints stage
# ===========================================================================

SUBJECT_CONSTRAINTS_CSV = WIKIDATA_DIR / "subject_constraints_types.csv"
VALUE_CONSTRAINTS_CSV = WIKIDATA_DIR / "value_constraints_types.csv"
SUBJECT_CONSTRAINTS_OUT_CSV = WICLEAN_DIR / "subject_constraints_types_clean.csv"
VALUE_CONSTRAINTS_OUT_CSV = WICLEAN_DIR / "value_constraints_types_clean.csv"

CLS_INST_COUNT_FILE = WICLEAN_DIR / "cls_inst_count.csv"
THRESHOLDS = {
    "avg_distance": 2,
    "ic_difference": -0.1,
    # "expansion_ratio": 0.5,
    # "depth_ratio": 0.5,
}

# ===========================================================================
#  Facts stage
# ===========================================================================

# ParseInstanceTypes
INST_TYPES_FOLDER = WICLEAN_DIR / "instTypes"
INST_META_MESSAGES_FILE = "wiki_instance_types_meta_messages.log"
INST_FACTS_FILE = "wiki_instance_types.tsv"
INST_TYPES_FILE = INST_TYPES_FOLDER / INST_FACTS_FILE

# ParseWikiFacts
FACTS_FOLDER = WICLEAN_DIR / "facts"
FACTS_META_MESSAGES_FILE = "wiki_facts_meta_messages.log"
FACTS_FILE = "wiki_facts.tsv"  # filename only; used by ParseWikiFacts

IDENTIFIERS_FILE = DATA_DIR / "external_identifier.csv"
NO_LABEL_INSTANCES_FILE = INST_TYPES_FOLDER / "no_label_instances.txt"
SCHOLARLY_ARTICLE_CLASSES_FILE = DATA_DIR / "scholarly_articles.csv"

# FactsTypeCheck
TYPECHECK_FACTS_PATH = FACTS_FOLDER / FACTS_FILE
TYPECHECK_OUTPUT_FOLDER = WICLEAN_DIR
TYPECHECK_META_MESSAGES_FILE = "wiclean_facts_constrainted_meta_messages.log"
TYPECHECK_FACTS_FILE = "wiclean_facts.tsv"

# ConstraintResimplification
RESIMP_FACTS_PATH = WICLEAN_DIR / TYPECHECK_FACTS_FILE
RESIMP_UNUSED_TYPES_CSV = WICLEAN_DIR / "constraints" / "rel_constraints_unused_types.csv"

# ===========================================================================
#  Analysis stage
# ===========================================================================

WICLEAN_FACTS_FILE = WICLEAN_DIR / "wiclean_facts.tsv"
STATS_OUTPUT_DIR = WICLEAN_DIR / "statistics/"
ROBUSTNESS_OUTPUT_FILE = STATS_OUTPUT_DIR / "robustness.txt"

# ===========================================================================
#  Ensure output directories exist
# ===========================================================================

for _dir in [INST_TYPES_FOLDER, FACTS_FOLDER,
             RESIMP_UNUSED_TYPES_CSV.parent,
             STATS_OUTPUT_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)
