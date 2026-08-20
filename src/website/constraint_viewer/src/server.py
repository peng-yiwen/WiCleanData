"""
Constraint Viewer — serve static assets and run relation summaries.

  GET  /api/models           → { "models": [ "mistral7b", ... ] }
  GET  /api/summary?model=&relation=  → metrics.compute_relation_analysis JSON
  POST /api/labels           → { "labels": { "Q5": "human", ... } } body: { "qids": ["Q5", ...] }

Taxonomy files `{model}_taxonomy.txt` and `{model}_mapping.txt` are discovered
from `constraint_viewer/data` only.

Optional model allowlist lives in `src/config.py` as `MODEL_ALLOWLIST`.
"""

from __future__ import annotations

import http.server
import json
import os
import sys
import traceback
import urllib.parse
from pathlib import Path
from config import APP_ROOT, DATA_DIR, MODEL_ALLOWLIST


def _model_name_allowlist() -> set[str] | None:
    if MODEL_ALLOWLIST is None:
        return None
    if len(MODEL_ALLOWLIST) == 0:
        return None
    return {x.strip() for x in MODEL_ALLOWLIST if x and x.strip()}


def _search_roots() -> list[Path]:
    return [DATA_DIR]


def list_models() -> list[str]:
    names: set[str] = set()
    for root in _search_roots():
        for path in root.glob("*_taxonomy.txt"):
            stem = path.name[: -len("_taxonomy.txt")]
            if (root / f"{stem}_mapping.txt").is_file():
                names.add(stem)
    allow = _model_name_allowlist()
    if allow is not None:
        names &= allow
    return sorted(names)


def resolve_model_paths(model: str) -> tuple[Path, Path]:
    for root in _search_roots():
        wikc = root / f"{model}_taxonomy.txt"
        mapping = root / f"{model}_mapping.txt"
        if wikc.is_file() and mapping.is_file():
            return wikc, mapping
    hint = ", ".join(str(r) for r in _search_roots())
    raise FileNotFoundError(
        f"Model {model!r} not found (need {model}_taxonomy.txt + {model}_mapping.txt in {hint})"
    )


class ConstraintViewerHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(APP_ROOT), **kwargs)

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path in ("/", "/index.html"):
            self.path = "/src/index.html"
            super().do_GET()
            return
        if parsed.path == "/json.html":
            self.path = "/src/json.html"
            super().do_GET()
            return
        if parsed.path == "/api/models":
            self._send_json({"models": list_models()})
            return
        if parsed.path == "/api/summary":
            self._handle_summary(parsed)
            return
        super().do_GET()

    def do_POST(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/labels":
            self._handle_labels_post()
            return
        self.send_error(404)

    def _handle_labels_post(self) -> None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        raw = self.rfile.read(length) if length > 0 else b"{}"
        try:
            body = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            self._send_json({"error": "Invalid JSON body"}, 400)
            return
        qids_in = body.get("qids")
        if not isinstance(qids_in, list):
            self._send_json({"error": "Body must include `qids` array"}, 400)
            return
        max_n = 4000
        import chunk as chunk_mod
        from metrics import load_wikc_labels

        wikc = load_wikc_labels()
        labels: dict[str, str] = {}
        for q in qids_in[:max_n]:
            qn = chunk_mod.normalize_qid(str(q).strip())
            if not (isinstance(qn, str) and qn.startswith("Q")):
                continue
            labels[qn] = wikc.get(qn, qn)
        self._send_json({"labels": labels})

    def _send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _handle_summary(self, parsed: urllib.parse.ParseResult) -> None:
        qs = urllib.parse.parse_qs(parsed.query)
        model = (qs.get("model") or [""])[0].strip()
        relation = (qs.get("relation") or [""])[0].strip().upper()
        if not model or not relation:
            self._send_json({"error": "Query parameters `model` and `relation` are required."}, 400)
            return
        if not relation.startswith("P") or not relation[1:].isdigit():
            self._send_json(
                {"error": "relation must be a Wikidata property id (e.g. P800)."},
                400,
            )
            return
        try:
            wikc_path, map_path = resolve_model_paths(model)
        except FileNotFoundError as e:
            self._send_json({"error": str(e)}, 404)
            return
        try:
            from metrics import compute_relation_analysis

            result = compute_relation_analysis(
                relation,
                str(wikc_path),
                str(map_path),
            )
            result["model"] = model
            self._send_json(result)
        except Exception as e:
            traceback.print_exc()
            self._send_json({"error": str(e)}, 500)


def main() -> None:
    port = int(os.environ.get("PORT", "8766"))
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            print(f"Invalid port {sys.argv[1]!r}; using {port}", file=sys.stderr)
    server = http.server.HTTPServer(("0.0.0.0", port), ConstraintViewerHandler)
    print(f"Constraint Viewer: http://127.0.0.1:{port}/  (serving {APP_ROOT})")
    server.serve_forever()


if __name__ == "__main__":
    main()
