# Wikidata

Raw Wikidata inputs for the cleaning pipeline: the truthy dump and SPARQL extracts.

## Layout

```
wikidata/
├── sparql/                          # SPARQL query templates (*.rq)
├── sample.nt                        # Small N-Triples sample for testing
├── subject_constraints_types.csv    # Subject-type constraints (from SPARQL)
├── value_constraints_types.csv      # Value-type constraints (from SPARQL)
├── latest-truthy.nt                 # Full dump (download; not committed)
└── ...                 
```

Additional CSVs produced by the queries in `sparql/` (e.g. `classes.csv`, `metaclasses.csv`) should also live in this directory `wikidata/`.

## Dump

Download the latest Wikidata truthy dump in N-Triples format into this folder:

```bash
wget https://dumps.wikimedia.org/wikidatawiki/entities/latest-truthy.nt.gz
gunzip latest-truthy.nt.gz
```

The pipeline expects `latest-truthy.nt` here (see `WIKIDATA_DUMP_FILE` in [`pipeline_config.py`](../../pipeline_config.py)).

## SPARQL queries

Queries live in [`sparql/`](sparql/). Run them against Wikidata (e.g. via [QLever](https://qlever.cs.uni-freiburg.de/wikidata)) and save each result as a CSV named after the query:
