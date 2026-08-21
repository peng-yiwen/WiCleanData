# Data

Input dumps, SPARQL extracts, and cleaned outputs for the WiCleanData pipeline. Override the root with `export WICLEAN_DATA_DIR=/path/to/your/data` (see [`pipeline_config.py`](../pipeline_config.py)).

## Layout

```
data/
├── wikidata/          # Raw Wikidata dump files
│   └── sparql/        # SPARQL queries for extracting constraints and classes
├── wikipedia/         # Mapping data between Wikidata and Wikipedia
└── wicleanData/       # Cleaned data produced by our pipeline
```

Run `make all` (or stage scripts under `src/`) after placing the required inputs under `wikidata/` and `wikipedia/`. (See corresponding README.md)