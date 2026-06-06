# constraint-viewer

Constraint summary demo with a single-page UI and JSON upload tab.

## Layout

- `data/` : all model and constraint data files
- `assets/` : front-end static assets (`app.js`, `style.css`)
- `src/` : application source code (`server.py`, `metrics.py`, `chunk.py`, html)

## Run

From this directory:

```bash
python3 src/server.py 8766
```

Then open:

- `http://127.0.0.1:8766/` (summary + upload tab)
- `http://127.0.0.1:8766/json.html` (redirects to upload tab)

## Model list behavior

- By default, models are discovered only from `data/`.
- Use `src/config.py` → `MODEL_ALLOWLIST` for an explicit model allowlist.
  - Example: `MODEL_ALLOWLIST = ["mistral7b"]`
  - Set `MODEL_ALLOWLIST = None` (or `[]`) to allow all discovered models.
