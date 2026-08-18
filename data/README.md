# Data

## Directory Structure

```
data/
├── wikidata/          # Raw Wikidata dump files
│   └── sparql/        # SPARQL queries for extracting constraints and properties
├── wikipedia/         # Mapping data between Wikidata and Wikipedia
└── wicleanData/       # Cleaned data produced by our pipeline
```

## Wikidata Dump

Download the latest Wikidata truthy dump in N-Triples format to `data/wikidata` from:
```bash
wget https://dumps.wikimedia.org/wikidatawiki/entities/latest-truthy.nt.gz
```

## SPARQL Queries

SPARQL queries are available in the `sparql/` folder. These can be used to extract constraint and property information from Wikidata. You can run these SPARQL queries using [QLever](https://qlever.cs.uni-freiburg.de/wikidata), a fast SPARQL engine that supports the full Wikidata knowledge base.
