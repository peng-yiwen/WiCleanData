# WiCleanData Pipeline
# ====================
# Execution order:  taxonomy → constraints → facts → analysis (optional)
#
# Usage:
#   make all             Run the full pipeline (taxonomy + constraints + facts)
#   make taxonomy        Run only the taxonomy cleaning stage
#   make constraints     Run constraints (automatically runs taxonomy first)
#   make facts           Run facts (automatically runs taxonomy + constraints first)
#   make analysis        Run evaluation analysis (requires completed pipeline)
#   make <stage> SKIP_DEPS=1   Run a single stage without triggering its dependencies

.PHONY: all taxonomy constraints facts analysis help

SKIP_DEPS ?= 0

all: facts
	@echo "\n✓ Full pipeline completed successfully."

# ---------------------------------------------------------------------------
#  Stage 1: Taxonomy cleaning
# ---------------------------------------------------------------------------
taxonomy:
	@echo "\n=========================================="
	@echo "  Stage 1/3: Taxonomy Cleaning"
	@echo "==========================================\n"
	cd src/taxonomy && bash run.sh

# ---------------------------------------------------------------------------
#  Stage 2: Constraints cleaning (depends on taxonomy)
# ---------------------------------------------------------------------------
ifeq ($(SKIP_DEPS),0)
constraints: taxonomy
else
constraints:
endif
	@echo "\n=========================================="
	@echo "  Stage 2/3: Constraints Cleaning"
	@echo "==========================================\n"
	cd src/constraints && bash run.sh

# ---------------------------------------------------------------------------
#  Stage 3: Facts cleaning (depends on constraints)
# ---------------------------------------------------------------------------
ifeq ($(SKIP_DEPS),0)
facts: constraints
else
facts:
endif
	@echo "\n=========================================="
	@echo "  Stage 3/3: Facts Cleaning"
	@echo "==========================================\n"
	cd src/facts && bash run.sh

# ---------------------------------------------------------------------------
#  Optional: Analysis / evaluation
# ---------------------------------------------------------------------------
ifeq ($(SKIP_DEPS),0)
analysis: facts
else
analysis:
endif
	@echo "\n=========================================="
	@echo "  Analysis & Evaluation"
	@echo "==========================================\n"
	cd src/analysis && bash run.sh

# ---------------------------------------------------------------------------
#  Help
# ---------------------------------------------------------------------------
help:
	@echo "WiCleanData Pipeline"
	@echo "===================="
	@echo ""
	@echo "Stages (run in order: taxonomy → constraints → facts):"
	@echo "  make all             Run full pipeline"
	@echo "  make taxonomy        Stage 1: Taxonomy cleaning"
	@echo "  make constraints     Stage 2: Constraints cleaning"
	@echo "  make facts           Stage 3: Facts cleaning"
	@echo "  make analysis        Optional: Run evaluation analysis"
	@echo ""
	@echo "Options:"
	@echo "  SKIP_DEPS=1          Skip dependency stages (run only the target stage)"
	@echo "                       e.g., make facts SKIP_DEPS=1"
	@echo ""
	@echo "Environment variables:"
	@echo "  WICLEAN_DATA_DIR     Override the default data directory"
	@echo ""
