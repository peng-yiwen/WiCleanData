# Taxonomy Viewer

A standalone webpage that visualizes the upward class hierarchy (taxonomy DAG) for a given class.

## Project Layout

```text
taxonomy_viewer/
  assets/
    index.html
    style.css
  data/
    OLD/                   
    cache/                  # generated/cached taxonomy JSON files
    manifest.json
    ...                     # taxonomy txt / mapping / label source files
  src/
    config.py               # centralized path config
    draw.py
    server.py
```

## Deploy

- First, install the required dependencies in **requirement.txt**
- Second, run `python3 src/server.py 8080`.
- Open the page in a browser.

## Usage

### Option 1: Sample Library
1. Select a **Model** from the dropdown, enter a **Class ID** (e.g. `Q515`), and click **Load**.
2. The file `data/cache/{model}_{Qid}.json` will be fetched and rendered.
3. If the file is not cached, the taxonomy will be generated on the fly.

### Option 2: Upload File
1. Switch to the **Upload File** tab.
2. Select a `.json` file from your local machine.
3. The DAG will render automatically.

## JSON Format

```json
{
  "nodes": {
    "<uri>": { "label": "Human-readable name", "url": "link or null", "isTopLevel": false }
  },
  "edges": [
    { "child": "<child-uri>", "parent": "<parent-uri>" }
  ],
  "query": {
    "uri": "<query-class-uri>",
    "label": "Class Name",
    "directSuperclasses": ["<parent-uri-1>", "<parent-uri-2>"]
  }
}
```

- **nodes** — each key is a URI; value has `label` (display text), `url` (clickable link or `null`), `isTopLevel` (`true` if the class is the root class in the taxonomy).
- **edges** — array of `{ child, parent }` pairs representing `rdfs:subClassOf` relationships.
- **query** (optional) — the class being queried. `directSuperclasses` lists the URIs of its immediate parent classes.



## Adding Data to the Sample Library

1. Place JSON files in `data/cache/` with the naming convention `{model}_{Qid}.json`.
2. Add new model names to `data/manifest.json` so they appear in the dropdown.

## Path Configuration

All main paths are centralized in `src/config.py`, including:

- project root
- `assets/` path
- `data/` path
- `data/cache/` path
- `data/manifest.json` path

## Node Colors

| Color | Meaning |
|-------|---------|
| Teal | Query class |
| Green tint | Direct superclass of the query class |
| Orange tint | Root class |
| Blue tint | Other class |


The DAG visualization component is adapted from the YAGO Knowledge Base website
(https://github.com/yago-naga/yago-website), licensed under Apache License 2.0.