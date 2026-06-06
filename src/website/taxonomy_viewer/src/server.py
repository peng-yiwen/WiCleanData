"""
Taxonomy Viewer — backend server.

Serves static files and provides /api/taxonomy?model=xxx&qid=Qxxx
that either returns a cached JSON from data/ or generates one on the fly
via draw.py.
"""
import http.server
import json
import os
import re
import sys
import traceback
from urllib.parse import urlparse, parse_qs

from config import ASSETS_DIR, CACHE_DIR, DATA_DIR, PROJECT_ROOT, ORIGINAL_MODEL

# ── lazy-loaded shared state ────────────────────────────────────────
_label_cache = {}   # label_key -> cls2label
_model_cache = {}   # model -> (adjacency, names, mapping, label_key)


def _get_labels(model):
    """Return cls2label dict. 'original' uses wikidatalabels.txt; others use wikclabels.txt."""
    label_key = ORIGINAL_MODEL if model == ORIGINAL_MODEL else "_default"
    if label_key not in _label_cache:
        if label_key == ORIGINAL_MODEL:
            from draw import load_original_label
            print("[server] Loading original Wikidata labels …")
            _label_cache[label_key] = load_original_label(data_dir=DATA_DIR)
        else:
            from draw import load_label
            print("[server] Loading class labels …")
            _label_cache[label_key] = load_label(data_dir=DATA_DIR)
        print(f"[server] Loaded {len(_label_cache[label_key])} labels ({label_key})")
    return _label_cache[label_key]


def _get_model_data(model):
    if model not in _model_cache:
        if model == ORIGINAL_MODEL:
            from draw import load_original_taxonomy
            print(f"[server] Loading original Wikidata taxonomy …")
            adjacency, names = load_original_taxonomy(data_dir=DATA_DIR)
            mapping = None
        else:
            from draw import load_taxonomy, load_mapping
            cls2label = _get_labels(model)
            print(f"[server] Loading taxonomy for model '{model}' …")
            adjacency, names = load_taxonomy(model, cls2label, data_dir=DATA_DIR)
            mapping = load_mapping(model, data_dir=DATA_DIR)
        _model_cache[model] = (adjacency, names, mapping)
        print(f"[server] Model '{model}' loaded ({len(names)} classes)")
    return _model_cache[model]


def generate_taxonomy(model, qid):
    """Return (json_data, warning) or (None, None)."""
    from draw import get_paths_root_to_target, generate_json_file

    cls2label = _get_labels(model)
    adjacency, names, mapping = _get_model_data(model)
    sub_adjacency, sub_names, resolved = get_paths_root_to_target(
        adjacency, names, qid, mapping
    )
    if sub_adjacency is None:
        return None, None
    warning = None
    if resolved != qid:
        qid_label = cls2label.get(qid, qid)
        resolved_label = cls2label.get(resolved, resolved)
        warning = f"{qid}({qid_label}) was merged to {resolved}({resolved_label})"
    json_data = generate_json_file(sub_adjacency, sub_names, cls2label, resolved)

    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, f"{model}_{qid}.json")
    with open(cache_path, "w") as f:
        json.dump(json_data, f, indent=2)

    return json_data, warning


# ── HTTP handler ────────────────────────────────────────────────────
class TaxonomyHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=PROJECT_ROOT, **kwargs)

    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path == "/api/taxonomy":
            self._handle_taxonomy(parsed)
        elif parsed.path in ("/", "/index.html"):
            self.path = "/assets/index.html"
            super().do_GET()
        elif parsed.path == "/style.css":
            self.path = "/assets/style.css"
            super().do_GET()
        else:
            super().do_GET()

    def _handle_taxonomy(self, parsed):
        params = parse_qs(parsed.query)
        model = params.get("model", [None])[0]
        qid = params.get("qid", [None])[0]

        if not model or not qid:
            self._json_error(400, "Both 'model' and 'qid' parameters are required.")
            return

        if not re.match(r"^Q\d+$", qid):
            self._json_error(400, f"Invalid qid format: {qid}. Expected Qnnn.")
            return

        cached = os.path.join(CACHE_DIR, f"{model}_{qid}.json")
        if os.path.isfile(cached):
            with open(cached, "r") as f:
                data = json.load(f)
            self._json_response(data, source="cache")
            return

        try:
            data, warning = generate_taxonomy(model, qid)
        except FileNotFoundError as e:
            self._json_error(404, str(e))
            return
        except Exception:
            self._json_error(500, traceback.format_exc())
            return

        if data is None:
            self._json_error(
                404,
                f"Class {qid} not found in the taxonomy of model '{model}'.",
            )
            return

        self._json_response(data, source="generated", warning=warning)

    def _json_response(self, data, source="cache", warning=None):
        body = json.dumps(data).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Source", source)
        if warning:
            self.send_header("X-Warning", warning)
        self.end_headers()
        self.wfile.write(body)

    def _json_error(self, code, message):
        body = json.dumps({"error": message}).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        sys.stderr.write(f"[{self.log_date_time_string()}] {fmt % args}\n")


# ── main ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    server = http.server.HTTPServer(("", port), TaxonomyHandler)
    print(f"[server] Taxonomy Viewer running at http://localhost:{port}/")
    print(f"[server] Assets directory: {ASSETS_DIR}")
    print(f"[server] Data directory: {DATA_DIR}")
    print(f"[server] Cache directory: {CACHE_DIR}")
    print(f"[server] Press Ctrl+C to stop.\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[server] Shutting down.")
        server.server_close()
