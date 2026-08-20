"""
Chunk type constraints via hierarchical clustering on taxonomy DAG distance.
Graph convention: top-down  (parent → child), single root.
"""

import csv
import time
import argparse
from collections import defaultdict, deque
from pathlib import Path
import os
from config import DATA_DIR, REL_SUBJECT_CSV, REL_VALUE_CSV, DEFAULT_TAXONOMY_ROOT_ID

import networkx as nx
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def normalize_qid(raw: str) -> str:
    """Normalize a Wikidata Q-id to ``Q`` + digits (e.g. ``q123`` / ``123`` → ``Q123``)."""
    s = str(raw).strip().upper()
    if s.startswith("Q") and len(s) > 1 and s[1:].isdigit():
        return s
    if s.isdigit():
        return f"Q{s}"
    return str(raw).strip()


def load_original_taxonomy(path: str) -> nx.DiGraph:
    """Load taxonomy as a top-down DAG: edges go parent → child."""
    G = nx.DiGraph()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) != 2:
                continue
            child, parent = normalize_qid(parts[0]), normalize_qid(parts[1])
            G.add_edge(parent, child)          # top-down
    return G



def load_comma_taxonomy(path: str) -> nx.DiGraph:
    """Load taxonomy from ``{llm}_taxonomy.txt``: comma-separated child,parent (Q-ids), top-down DAG."""
    G = nx.DiGraph()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) != 2:
                continue
            child, parent = normalize_qid(parts[0]), normalize_qid(parts[1])
            G.add_edge(parent, child)
    return G


def load_comma_mapping(path: str) -> dict[str, str]:
    """Load merge map from ``{llm}_mapping.txt``: comma-separated original,replacement (Q-ids)."""
    mapping: dict[str, str] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) != 2:
                continue
            orig, new = normalize_qid(parts[0]), normalize_qid(parts[1])
            mapping[orig] = new
    return mapping




def _normalize_property_id(property_id: str) -> str:
    return property_id.strip().upper()


def load_relation_constraint_types(
    subject_csv: str,
    value_csv: str,
    property_id: str,
):
    """Load subject-side and object-side types for a Wikidata property.

    Rows are matched on the ``property`` column against *subject_csv*
    (``subject_type`` / ``subject_type_label``) and *value_csv*
    (``value_type`` / ``value_type_label``).

    Returns:
        (subject_types, object_types): each a list of (qid, label), order-preserving
        with first occurrence kept when the same QID appears on multiple rows.
    """
    prop = _normalize_property_id(property_id)

    def _dedupe_rows(rows: list[tuple[str, str]]) -> list[tuple[str, str]]:
        seen: set[str] = set()
        out: list[tuple[str, str]] = []
        for qid, lab in rows:
            if qid in seen:
                continue
            seen.add(qid)
            out.append((qid, lab))
        return out

    subjects: list[tuple[str, str]] = []
    with open(subject_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if _normalize_property_id(row.get("property", "")) != prop:
                continue
            subjects.append((normalize_qid(row["class"]), row["classLabel"]))

    objects: list[tuple[str, str]] = []
    with open(value_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if _normalize_property_id(row.get("property", "")) != prop:
                continue
            objects.append((normalize_qid(row["class"]), row["classLabel"]))

    return _dedupe_rows(subjects), _dedupe_rows(objects)


def load_property_label(subject_csv: str, value_csv: str, property_id: str) -> str | None:
    """Return ``property_label`` from the first CSV row matching *property_id* (subject then value file)."""
    prop = _normalize_property_id(property_id)
    for path in (subject_csv, value_csv):
        with open(path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if _normalize_property_id(row.get("property", "")) != prop:
                    continue
                lab = (row.get("propertyLabel") or "").strip()
                if lab:
                    return lab
    return None



# ---------------------------------------------------------------------------
# DAG distance via LCA
# ---------------------------------------------------------------------------

def dag_distance(u, v, G):
    """
    Shortest distance between u and v through any common ancestor.
    min over all common ancestors a of  dist(a→u) + dist(a→v).
    """
    if u == v:
        return 0
    anc_u = nx.ancestors(G, u) | {u}
    anc_v = nx.ancestors(G, v) | {v}
    common = anc_u & anc_v
    if not common:
        return float("inf")

    best = float("inf")
    for a in common:
        du = nx.shortest_path_length(G, a, u)
        dv = nx.shortest_path_length(G, a, v)
        best = min(best, du + dv)
    return best


# ---------------------------------------------------------------------------
# Hierarchical Clustering
# ---------------------------------------------------------------------------

def compute_distance_matrix(types_qids, G):
    n = len(types_qids)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = dag_distance(types_qids[i], types_qids[j], G)
            dist[i, j] = d
            dist[j, i] = d
    return dist


def threshold_median(dist):
    n = dist.shape[0]
    return float(np.median(dist[np.triu_indices(n, k=1)]))


def threshold_max_gap(Z):
    """Cut at the largest jump in successive merge distances."""
    merge_dists = Z[:, 2]
    if len(merge_dists) <= 1:
        return merge_dists[0]
    gaps = np.diff(merge_dists)
    idx = int(np.argmax(gaps))
    # return (merge_dists[idx] + merge_dists[idx + 1]) / 2.0
    if idx+1 >= len(merge_dists) - 1:
        return merge_dists[idx]
    return merge_dists[idx+1]


def chunk_by_hierarchical_clustering(types_qids, dist, method="average", threshold_mode="median"):
    n = len(types_qids)
    # if there is only one type, return it as a single chunk
    if n <= 1:
        return [types_qids], None, None

    # perform hierarchical clustering on the distance matrix
    Z = linkage(squareform(dist), method=method) # method: centroid / median / ward

    if threshold_mode == "median":
        threshold = threshold_median(dist)
    elif threshold_mode == "max_gap":
        threshold = threshold_max_gap(Z)
    else:
        raise ValueError(f"Unknown threshold_mode: {threshold_mode}")

    labels = fcluster(Z, t=threshold, criterion="distance")
    groups = defaultdict(list)
    for idx, lbl in enumerate(labels):
        groups[lbl].append(types_qids[idx])
    return list(groups.values()), Z, threshold


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_labels(type_list, qid_to_label):
    return [f"{qid} ({qid_to_label.get(qid, qid)})" for qid in type_list]


def resolve_nodes(
    nodes,
    G_clean,
    mapping,
    verbose: bool = True,
    return_mapping_hits: bool = False,
):
    """Step 1: keep only nodes present in G_clean, resolving merges via mapping.

    Returns:
        resolved (list): deduplicated list of nodes that exist in G_clean.
        dropped  (list): nodes absent from both G_clean and mapping.
    """
    resolved, dropped, seen = [], [], set()
    mapping_hits: list[tuple[str, str]] = []
    for n in nodes:
        target = n
        while target in mapping:
            # print(f"  Resolving {target} -> {mapping.get(target, target)}")
            target = mapping.get(target, target)
        if target != n and verbose:
            print(f"    - Found mapping for {n} -> {target}")
        if target != n:
            mapping_hits.append((n, target))
        if target in G_clean:
            if target not in seen:
                seen.add(target)
                resolved.append(target)
        else:
            dropped.append(n)
    if return_mapping_hits:
        return resolved, dropped, mapping_hits
    return resolved, dropped


def remove_redundant(nodes, G_clean):
    """Step 2: if a child and one of its ancestors are both in *nodes*,
    drop the child (the ancestor already covers it).

    Returns:
        reduced (list): non-redundant subset of *nodes*.
    """
    node_set = set(nodes)
    redundant = set()
    for n in nodes:
        ancestors = nx.ancestors(G_clean, n)
        if ancestors & (node_set - {n}):
            redundant.add(n)
    return [n for n in nodes if n not in redundant]


def print_chunks(name, chunks, qid_to_label):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    for i, chunk in enumerate(chunks):
        print(f"  Chunk {i+1}: {get_labels(chunk, qid_to_label)}")
    print(f"  Total chunks: {len(chunks)}")

# ---------------------------------------------------------------------------
# Lowest Common Ancestor
# ---------------------------------------------------------------------------


def _reverse_bfs_lcas(nodes, G_clean):
    """Step 3: bottom-up BFS that finds **merge-event** LCAs only.

    Starting from *nodes*, propagate upward one hop at a time.  An ancestor is
    recorded as an LCA only when its covered set is **not** a subset of any
    already-recorded LCA's covered set — i.e. it represents a genuinely new
    merge of previously disjoint groups.

    Returns:
        dict: {lca_node: [covered_node_set, max_hop_distance], ...}
              ordered by ascending distance.  Each entry represents one
              merge event; no entry's covered set is a subset of another's
              that was found at a smaller distance.
    """
    node_set = set(nodes)

    if len(node_set) <= 1:
        node = next(iter(node_set))
        return {node: [node_set.copy(), 0]}

    # coverage[v] = {original_node: min_hops_from_original_node_to_v}
    coverage = defaultdict(dict)
    queue = deque()
    # Re-enqueue a node when its coverage *set* grows (not just when
    # distance improves).  Each node can be enqueued at most |node_set|
    # times, so total work is O(|E| * |node_set|).
    enqueued_cov_size = defaultdict(int)

    for n in node_set:
        coverage[n][n] = 0
        queue.append(n)
        enqueued_cov_size[n] = 1

    lca_results = {}

    while queue:
        current = queue.popleft()

        covered = set(coverage[current].keys())

        if len(covered) >= 2:
            max_dist = max(coverage[current].values())
            if current not in lca_results or len(covered) > len(lca_results[current][0]):
                lca_results[current] = [covered.copy(), max_dist]

        for parent in G_clean.predecessors(current):
            updated = False
            for orig_node, orig_dist in coverage[current].items():
                new_dist = orig_dist + 1
                if orig_node not in coverage[parent] or coverage[parent][orig_node] > new_dist:
                    coverage[parent][orig_node] = new_dist
                    updated = True

            if updated:
                new_size = len(coverage[parent])
                if new_size > enqueued_cov_size[parent]:
                    enqueued_cov_size[parent] = new_size
                    queue.append(parent)

    # Filter: discard an LCA when another LCA has the exact same covered
    # set and either (a) strictly shorter distance, or (b) same distance
    # but the other is a descendant (i.e. this node is the ancestor).
    filtered = {}
    for lca, (covered, dist) in lca_results.items():
        dominated = False
        for other, (o_cov, o_dist) in lca_results.items():
            if other == lca or o_cov != covered:
                continue
            if o_dist < dist:
                dominated = True
                break
            if o_dist == dist and nx.has_path(G_clean, lca, other):
                dominated = True
                break
        if not dominated:
            filtered[lca] = [covered, dist]

    return filtered


def _pairwise_lcas(nodes, G_clean):
    """Find LCAs by enumerating all pairs and computing their LCAs in the DAG.

    Steps:
      1. For every pair (u, v), find all common ancestors, keep only the
         *lowest* ones (no descendant is also a common ancestor).
      2. For each LCA candidate, compute which original nodes it covers
         (i.e. are descendants of it) and the max shortest-path distance.
      3. Apply the same merge-event filter as _reverse_bfs_lcas.

    Returns:
        dict  {lca_node: [covered_node_set, max_hop_distance], ...}
    """
    node_set = set(nodes)
    node_list = list(node_set)

    if len(node_set) <= 1:
        node = next(iter(node_set))
        return {node: [node_set.copy(), 0]}

    # --- Step 1: collect LCA candidates from all pairs ---
    # nx.lowest_common_ancestor returns one LCA per pair.
    # For DAGs with multiple LCAs per pair, we fall back to manual search.
    lca_candidates = set()
    for i in range(len(node_list)):
        for j in range(i + 1, len(node_list)):
            u, v = node_list[i], node_list[j]
            anc_u = nx.ancestors(G_clean, u) | {u}
            anc_v = nx.ancestors(G_clean, v) | {v}
            common = anc_u & anc_v
            if not common:
                continue
            # keep only lowest: no proper descendant is also in common
            for a in common:
                if not (nx.descendants(G_clean, a) & common):
                    lca_candidates.add(a)

    # --- Step 2: compute covered set & max distance for each candidate ---
    lca_results = {}
    for lca in lca_candidates:
        lca_desc = nx.descendants(G_clean, lca) | {lca}
        covered = set()
        max_dist = 0
        for n in node_set:
            if n in lca_desc:
                d = nx.shortest_path_length(G_clean, lca, n)
                covered.add(n)
                max_dist = max(max_dist, d)
        if len(covered) >= 2:
            lca_results[lca] = [covered, max_dist]

    # --- Step 3: only discard an LCA when another has the exact same
    # covered set but a strictly shorter distance. ---
    filtered = {}
    for lca, (covered, dist) in lca_results.items():
        dominated = False
        for other, (o_cov, o_dist) in lca_results.items():
            if other != lca and o_cov == covered and o_dist < dist:
                dominated = True
                break
        if not dominated:
            filtered[lca] = [covered, dist]

    return filtered


def compare_lca_methods(nodes, G_clean, qid_to_label=None):
    """Run both LCA algorithms and print a side-by-side comparison."""
    t0 = time.time()
    res_bfs = _reverse_bfs_lcas(nodes, G_clean)
    t_bfs = time.time() - t0

    t0 = time.time()
    res_pair = _pairwise_lcas(nodes, G_clean)
    t_pair = time.time() - t0

    def _fmt(results):
        lines = []
        for lca, (covered, dist) in sorted(results.items(), key=lambda kv: kv[1][1]):
            cov_str = ", ".join(sorted(covered))
            label = f" ({qid_to_label[lca]})" if qid_to_label and lca in qid_to_label else ""
            lines.append(f"    {lca}{label}  dist={dist}  covers=[{cov_str}]")
        return "\n".join(lines) if lines else "    (none)"

    match = (set(res_bfs.keys()) == set(res_pair.keys())
             and all(set(res_bfs[k][0]) == set(res_pair[k][0])
                     and res_bfs[k][1] == res_pair[k][1]
                     for k in res_bfs))

    print(f"  BFS LCAs result ({len(res_bfs)} LCAs, {t_bfs:.4f}s):\n{_fmt(res_bfs)}")
    # print(f"  Pairwise result ({len(res_pair)} LCAs, {t_pair:.4f}s):\n{_fmt(res_pair)}")
    # print(f"  Match: {'YES' if match else '*** NO ***'}")

    if not match:
        only_bfs = set(res_bfs.keys()) - set(res_pair.keys())
        only_pair = set(res_pair.keys()) - set(res_bfs.keys())
        if only_bfs:
            print(f"  Only in BFS: {only_bfs}")
        if only_pair:
            print(f"  Only in Pairwise: {only_pair}")
        common_keys = set(res_bfs.keys()) & set(res_pair.keys())
        for k in common_keys:
            if set(res_bfs[k][0]) != set(res_pair[k][0]) or res_bfs[k][1] != res_pair[k][1]:
                print(f"  Diff at {k}: BFS={res_bfs[k]} vs Pair={res_pair[k]}")

    return match


def check_cover_all_lcas():
    pass
# 

# ---------------------------------------------------------------------------
# Main (standalone test)
# ---------------------------------------------------------------------------

def _constraints_dir() -> Path:
    """Data directory for constraint-viewer."""
    return DATA_DIR


def _default_rel_constraint_paths() -> tuple[str, str]:
    return str(REL_SUBJECT_CSV), str(REL_VALUE_CSV)


def resolve_llm_taxonomy_paths(llm: str, data_dir: Path | None) -> tuple[Path, Path]:
    """``{llm}_taxonomy.txt`` and ``{llm}_mapping.txt`` under *data_dir* (default: data/)."""
    base = data_dir if data_dir is not None else DATA_DIR
    return base / f"{llm}_taxonomy.txt", base / f"{llm}_mapping.txt"


def run_relation_side(
    side_title: str,
    type_list: list[tuple[str, str]],
    G: nx.DiGraph,
    mapping: dict,
    compare_lca: bool,
):
    """Run clustering + optional LCA comparison for one side (subject or object)."""
    print(f"\n{side_title}")
    print("=" * len(side_title))
    if not type_list:
        print("  (no types for this property in the corresponding constraints file)")
        return

    types_qids = [t[0] for t in type_list]
    qid_to_label = {t[0]: t[1] for t in type_list}
    print(f"  {len(types_qids)} types: {get_labels(types_qids, qid_to_label)}")

    valid_nodes, dropped = resolve_nodes(types_qids, G, mapping)
    if not valid_nodes:
        print("  No valid nodes remaining after resolving.")
        return
    if dropped:
        print(f"  Dropped Types not in taxonomy: {dropped}")
    types_qids = remove_redundant(valid_nodes, G)
    if set(types_qids) != set(valid_nodes):
        print("  Pruned nodes: ", set(valid_nodes) - set(types_qids))

    print("\n  Computing pairwise DAG distances ...")
    dist = compute_distance_matrix(types_qids, G)
    for mode in ("median", "max_gap"):
        chunks, Z, thr = chunk_by_hierarchical_clustering(types_qids, dist, threshold_mode=mode)
        print(f"\n  --- threshold = {mode} ({thr:.2f}) ---")
        print_chunks(f"Hierarchical – {mode}", chunks, qid_to_label)
        if compare_lca:
            for i, c in enumerate(chunks):
                print(f"\n  Chunk {i+1}: {get_labels(c, qid_to_label)}")
                compare_lca_methods(c, G, qid_to_label)


def main():
    parser = argparse.ArgumentParser(
        description="Cluster relation type constraints; taxonomy from {llm}_taxonomu.txt / {llm}_mapping.txt.",
    )
    parser.add_argument(
        "--relation",
        default="P1000",
        help="Wikidata property id (e.g. P1000).",
    )
    parser.add_argument(
        "--llm",
        default="mistral7b",
        help="Base name for {llm}_taxonomy.txt and {llm}_mapping.txt (comma-separated Q-ids, same as taxonomy_viewer).",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Directory containing {llm}_taxonomy.txt and {llm}_mapping.txt (default: directory of chunk.py).",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve() if args.data_dir else None
    wikc_path, map_path = resolve_llm_taxonomy_paths(args.llm, data_dir)
    if not wikc_path.is_file():
        raise FileNotFoundError(f"Taxonomy not found: {wikc_path}")
    if not map_path.is_file():
        raise FileNotFoundError(f"Mapping not found: {map_path}")

    print(f"Loading taxonomy from {wikc_path} ...")
    G = load_comma_taxonomy(str(wikc_path))
    mapping = load_comma_mapping(str(map_path))

    default_subj, default_val = _default_rel_constraint_paths()
    if not os.path.isfile(default_subj) or not os.path.isfile(default_val):
        raise FileNotFoundError(
            f"Expected relation constraint CSVs next to chunk.py:\n  {default_subj}\n  {default_val}"
        )
    print(f"Property {args.relation}: loading types from\n  {default_subj}\n  {default_val}")
    subject_types, object_types = load_relation_constraint_types(
        default_subj, default_val, args.relation
    )

    run_relation_side("Subject:", subject_types, G, mapping, compare_lca=True)
    run_relation_side("Object:", object_types, G, mapping, compare_lca=True)


if __name__ == "__main__":
    main()
