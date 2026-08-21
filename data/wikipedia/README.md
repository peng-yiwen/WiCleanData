# Wikipedia

Wikipedia↔Wikidata page mappings used to filter and align the cleaned taxonomy.

## Layout

```
wikipedia/
├── mapping.ipynb    # Convert wbc_entity_usage SQL dumps → mapping text files
├── enwiki           # Mapping for English Wikipedia (example / LFS)
├── frwiki           # … (add other languages the same way)
└── …
```

## Download

Get the `wbc_entity_usage` SQL dumps from the [Wikimedia dumps site](https://dumps.wikimedia.org/enwiki/) (change the language code in the URL for other wikis).

In our experiments we use: **enwiki**, **frwiki**, **dewiki**, **zhwiki**, **arwiki**, **eswiki**.

## Build mappings

[`mapping.ipynb`](mapping.ipynb) parses a dump such as `enwiki-YYYYMMDD-wbc_entity_usage.sql` and writes a plain-text mapping file named after the wiki (e.g. `enwiki`).

Each line is:

```
wikipediaID,wikidataID,language
```

Example:

```
12345,Q42,enwiki
```

Place one file per language in this directory; the pipeline reads them from `data/wikipedia/`.