import os

# Project root: wicleanData/
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

# Data directories
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
WIKIDATA_DIR = os.path.join(DATA_DIR, "wikidata")
WIKIPEDIA_DIR = os.path.join(WIKIDATA_DIR, "wikipedia")

# Input files (from SPARQL / Wikidata dump)
WIKIDATA_DUMP_FILE = os.path.join(WIKIDATA_DIR, "latest-truthy.nt")
CLASSES_FILE = os.path.join(WIKIDATA_DIR, "classes.csv")
METACLASSES_FILE = os.path.join(WIKIDATA_DIR, "metaclasses.csv")
CLS_INST_COUNT_FILE = os.path.join(WIKIDATA_DIR, "class_instance_count.csv")
BFO_CLASSES_FILE = os.path.join(WIKIDATA_DIR, "bfo_classes.csv")

# Init taxonomy files (produced by extractTaxonomy.py)
TAXONOMY_FILE = os.path.join(WIKIDATA_DIR, "wiki_taxonomy_extracted.tsv")
TAXONOMY_LABELS_FILE = os.path.join(WIKIDATA_DIR, "wiki_taxonomy_extracted_labels.tsv")
TAXONOMY_DESCRIPTIONS_FILE = os.path.join(WIKIDATA_DIR, "wiki_taxonomy_extracted_descriptions.tsv")
EMBEDDING_PKL_FILE = os.path.join(WIKIDATA_DIR, "wiki_2026_labels_emb.pkl")

# Results directories
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
LLM_OUTPUT_DIR = os.path.join(RESULTS_DIR, "llm_output")
INTERMEDIATE_GRAPHS_DIR = os.path.join(RESULTS_DIR, "intermediate_graphs")

# Prompts
PROMPTS_DIR = os.path.join(PROJECT_ROOT, "prompts")
SUBCLASS_EVAL_PROMPT = os.path.join(PROMPTS_DIR, "SubClassEval.txt")

# Final output
WICLEAN_OUTPUT_DIR = os.path.join(DATA_DIR, "wicleanData")
WICLEAN_TAXONOMY_FILE = os.path.join(WICLEAN_OUTPUT_DIR, "wicleanTaxonomy.txt")
WICLEAN_MAPPING_FILE = os.path.join(WICLEAN_OUTPUT_DIR, "wiclean_mapping.txt")

# LLM prediction file patterns (relative filenames within LLM_OUTPUT_DIR)
MAJORITY_PREDICTIONS_FILE = "llm_majority_predictions.txt"
MAJORITY_REWIRE_LINKS_FILE = "majority_rewire_links.txt"
MAJORITY_PREDICTIONS_REWIRE_FILE = "majority_predictions_rewire.json"

# Model list
LLM_MODELS = ['mistral24b', 'gemma27b', 'qwen32b']
