import os

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)

ASSETS_DIR = os.path.join(PROJECT_ROOT, "assets")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CACHE_DIR = os.path.join(DATA_DIR, "cache")
# ORIGINAL_DIR = os.path.join(DATA_DIR, "OLD")
MANIFEST_PATH = os.path.join(DATA_DIR, "manifest.json")
ORIGINAL_MODEL = "wikidata_2026"
