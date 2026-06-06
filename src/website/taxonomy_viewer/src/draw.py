import warnings
warnings.filterwarnings("ignore")
from sknetwork.path import breadth_first_search
from scipy.sparse import csr_matrix
import numpy as np
import json
import os
import sys

from config import CACHE_DIR

ROOT_CLASS = "Q35120"


def make_uri(qid):
    return f"http://www.wikidata.org/entity/{qid}"


def make_url(qid):
    return f"https://www.wikidata.org/wiki/{qid}"


def load_label(label_path="wikclabels_2026.txt", data_dir=None):
    if data_dir:
        label_path = os.path.join(data_dir, label_path)
    cls2label = {}
    with open(label_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) > 1:
                cls2label[parts[0]] = parts[1][1:-1] # be careful
    return cls2label


def load_taxonomy(model, cls2label, data_dir=None):
    filepath = f'{model}_wikc.txt'
    if data_dir:
        filepath = os.path.join(data_dir, filepath)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f'{filepath} not found')

    edges = []
    nodes_set = set()
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            child, parent = line.split(',')
            edges.append((child, parent))
            nodes_set.add(child)
            nodes_set.add(parent)

    names = sorted(nodes_set)
    name_to_idx = {name: i for i, name in enumerate(names)}

    rows, cols = [], []
    for child, parent in edges:
        rows.append(name_to_idx[child])
        cols.append(name_to_idx[parent]) # subclass_of relation

    n = len(names)
    adjacency = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))

    # check all classes have labels
    missing = [name for name in names if name not in cls2label]
    if missing:
        # raise ValueError(f"Warning: {len(missing)} classes missing labels: {missing[:5]}...")
        print(f"Warning: {len(missing)} classes missing labels: {missing[:5]}...")
    
    # labels = [cls2label[name] for name in names]
    return adjacency, names


def get_paths_root_to_target(adjacency, names, target, mapping=None):

    resolved = target
    if resolved not in names:
        while resolved not in names:
            if mapping is None or resolved not in mapping:
                print(f"Node {target} not in the taxonomy or no mapping files found.")
                return None, None, target
            resolved = mapping[resolved]
        print(f"{target} is merged to {resolved}")

    target_index = names.index(resolved)
    ancestors = breadth_first_search(adjacency, source=target_index)
    sub_adjacency = adjacency[ancestors, :][:, ancestors]
    sub_names = list(np.array(names)[ancestors])

    return sub_adjacency, sub_names, resolved


def generate_json_file(sub_adjacency, sub_names, cls2label, target):
    nodes = {}
    for qid in sub_names:
        uri = make_uri(qid)
        nodes[uri] = {
            "label": cls2label.get(qid, qid),
            "url": make_url(qid),
            "isTopLevel": qid == ROOT_CLASS,
        }

    edges = []
    cx = sub_adjacency.tocoo()
    for i, j in zip(cx.row, cx.col):
        edges.append({
            "child": make_uri(sub_names[i]),
            "parent": make_uri(sub_names[j]),
        })

    target_idx = sub_names.index(target)
    direct_parents = [
        make_uri(sub_names[idx]) for idx in sub_adjacency[target_idx].indices
    ]

    result = {
        "query": {
            "uri": make_uri(target),
            "label": cls2label.get(target, target),
            "directSuperclasses": direct_parents,
        },
        "nodes": nodes,
        "edges": edges,
    }
    return result


## ── original wikidata ──────────────────────────────────────────────

def load_original_label(data_dir=None):
    """Load labels from wikidatalabels.txt (full Wikidata label set)."""
    filepath = "wikidata_2026_labels.txt" # TBC
    if data_dir:
        filepath = os.path.join(data_dir, filepath)
    cls2label = {}
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) > 1:
                cls2label[parts[0]] = parts[1][1:-1] # be careful
    return cls2label


def load_original_taxonomy(data_dir=None):
    """Load the original Wikidata taxonomy from wikidata.txt.
    Missing labels are allowed — qid is used as fallback."""
    filepath = "wikidata_2026.txt" # TBc
    if data_dir:
        filepath = os.path.join(data_dir, filepath)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f'{filepath} not found')

    edges = []
    nodes_set = set()
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            child, parent = line.split(',')
            edges.append((child, parent))
            nodes_set.add(child)
            nodes_set.add(parent)

    names = sorted(nodes_set)
    name_to_idx = {name: i for i, name in enumerate(names)}

    rows, cols = [], []
    for child, parent in edges:
        rows.append(name_to_idx[child])
        cols.append(name_to_idx[parent])

    n = len(names)
    adjacency = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))

    return adjacency, names


def load_mapping(model, data_dir=None):
    filepath = f"{model}_mapping.txt"
    if data_dir:
        filepath = os.path.join(data_dir, filepath)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"{filepath} not found")
    mapping = {}
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            original, new = line.split(',')
            mapping[original] = new
    return mapping


if __name__ == "__main__":
    # parameters
    model = sys.argv[1] if len(sys.argv) > 1 else "gemma4b"
    target = sys.argv[2] if len(sys.argv) > 2 else "Q515"

    # load data
    if model == "wikidata_2026": # TBC
        cls2label = load_original_label()
        adjacency, names = load_original_taxonomy()
    else:
        cls2label = load_label("wikclabels_2026.txt")
        adjacency, names = load_taxonomy(model, cls2label)
        mapping = load_mapping(model)
    sub_adjacency, sub_names, resolved = get_paths_root_to_target(adjacency, names, target, mapping)

    if sub_adjacency is not None:
        json_data = generate_json_file(sub_adjacency, sub_names, cls2label, resolved)
        os.makedirs(CACHE_DIR, exist_ok=True)
        output_file = os.path.join(CACHE_DIR, f"{model}_{target}.json")
        with open(output_file, 'w') as f:
            json.dump(json_data, f, indent=2)
        print(f"Generated {output_file}")
