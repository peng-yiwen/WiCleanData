"""
Metrics to evaluate whether an LCA is a good replacement for a chunk of types.

Graph convention: top-down DAG (parent → child), with a single root.

Metrics:
  1. avg_distance    – Average distance from LCA down to each type in the chunk
  2. expansion_ratio – |descendants(LCA)| / sum(|descendants(t)| for t in chunk)
  3. depth_ratio     – depth(LCA) / avg_depth(chunk types)
  4. ic_ratio        – IC(LCA) / avg IC(chunk types)
  5. in_wikc         – Whether the LCA appears in the wikc_plus (cleaned Wikidata) taxonomy
"""

import math
import networkx as nx
from collections import defaultdict
import chunk
import argparse
from pathlib import Path

# ---------------------------------------------------------------------------
# Depth helper  (root → node shortest path length)
# ---------------------------------------------------------------------------

def get_cls_instance_count(file_loc):
    cls_instance_count = dict()
    with open(file_loc, 'r') as f:
        for line in f:
            cls, count = line.strip().split('\t')
            cls_instance_count[cls] = int(count)
    return cls_instance_count


def getSubClasses(cls, classes, taxonomyDown):
    """Adds all subclasses of a class <cls> (including <cls>) to the set <classes>"""
    classes.add(cls)
    # Make a check before because it's a defaultdict,
    # which would create cls if it's not there
    if cls in taxonomyDown:
        for sc in taxonomyDown[cls]:
            getSubClasses(sc, classes, taxonomyDown)

def getDescendants(cls, taxonomyDown):
    """Returns the set of all child classes of <cls> (including <cls>)"""
    classes=set()
    getSubClasses(cls, classes, taxonomyDown)  
    return classes


def cumulative_stats_for_class(cls, stats, taxonomyDown):
    """Cumulative statistics of classes
    Args:
        cls (str): class to be calculated.
        stats (dict): dict of direct instances count for each class.
        taxonomyDown (dict): Taxonomy from top to down. -> here consider the clean taxonomy
    """
    descendants = getDescendants(cls, taxonomyDown) # including cls itself
    return sum(stats.get(descendant, 0) for descendant in descendants)


def get_depth(node, G, root='Q35120'):
    if root is None:
        raise ValueError("Root is required")
    return nx.shortest_path_length(G, root, node)


def load_clean_taxonomy(file_loc):
    cleanWikiTaxonDown = defaultdict(set)
    with open(file_loc, 'r') as clean:
        for line in clean:
            child, parent = line.strip().split('\t')
            cleanWikiTaxonDown[parent[3:]].add(child[3:]) # remove wd: prefix
    return nx.DiGraph(cleanWikiTaxonDown)


# ---------------------------------------------------------------------------
# Metric 1: Average Distance (LCA → each type)
# ---------------------------------------------------------------------------

def metric_avg_distance(lca, chunk_types, G):
    """
    Average distance from LCA down to each type in the chunk.
    G: clean taxonomy graph
    Chunk types: the updated list of classes
    """
    if lca is None:
        raise ValueError("A valid LCA is required")
    total = 0
    for t in chunk_types:
        total += nx.shortest_path_length(G, lca, t)
    return total / len(chunk_types)


# ---------------------------------------------------------------------------
# Metric 2: Expansion Ratio
# ---------------------------------------------------------------------------

def metric_expansion_ratio(lca, chunk_types, G):
    """
    |descendants(LCA)| / sum(set(descendants(t) for t in chunk)).
    1.0 = no expansion;  >>1 = LCA covers far more than original types.
    G: clean taxonomy graph
    """
    if lca is None:
        raise ValueError("A valid LCA is required")

    lca_descendants = set(nx.descendants(G, lca)).union({lca})
    lca_count = len(lca_descendants)
    ori_classes = set()
    for t in chunk_types:
        ori_classes.update(set(nx.descendants(G, t)))
        ori_classes.add(t)
    original_count = len(ori_classes)

    # check 
    included = ori_classes - lca_descendants
    assert len(included) == 0, "Error in Expansion Ratio: some types are not included in the LCA"
    return lca_count / original_count


# ---------------------------------------------------------------------------
# Metric 3: Depth Ratio
# ---------------------------------------------------------------------------

def metric_depth_ratio(lca, chunk_types, G, root='Q35120'):
    """
    Depth(LCA) / avg_depth(chunk types).
    G: clean taxonomy graph
    """
    if lca is None:
        raise ValueError("A valid LCA is required")
    lca_d = get_depth(lca, G, root)
    avg_d = metric_avg_distance(lca, chunk_types, G) + lca_d 
    return lca_d / avg_d


# ---------------------------------------------------------------------------
# Metric 4: IC Ratio
# ---------------------------------------------------------------------------

def metric_ic_difference(lca, chunk_types, G, cls_inst_count):
    """IC(LCA) - IC(chunk_types_union).

    IC(x) = -log(cumulative_instances(x) / N).

    To avoid double-counting instances shared by overlapping chunk_type
    subtrees, we compute the union count as:
      cumul(LCA) - sum(direct_count(c) for c in between_classes)
    where between_classes = descendants(LCA) \\ union(descendants(t) for t in chunk_types).
    """
    if lca is None:
        raise ValueError("A valid LCA is required")
    N = sum(cls_inst_count.values())

    taxonomyDown = nx.to_dict_of_lists(G)

    lca_desc = getDescendants(lca, taxonomyDown) # including lca itself
    c_lca = sum(cls_inst_count.get(d, 0) for d in lca_desc)

    chunk_desc = set()
    for t in chunk_types:
        chunk_desc.update(getDescendants(t, taxonomyDown))

    between_classes = lca_desc - chunk_desc
    c_sub = c_lca - sum(cls_inst_count.get(c, 0) for c in between_classes)

    ic_lca = -math.log(c_lca / N) if 0 < c_lca < N else 0.0
    ic_sub = -math.log(c_sub / N) if 0 < c_sub < N else 0.0

    if ic_sub == 0:
        return 1.0 if ic_lca == 0 else 0.0
    return ic_lca - ic_sub



# ---------------------------------------------------------------------------
# Evaluate all metrics for one chunk
# ---------------------------------------------------------------------------

def evaluate_chunk(lca, chunk_types, G, root='Q35120'):

    # load cls_inst_count
    cls_inst_count = get_cls_instance_count('cls_inst_count.txt')
    # metrics
    avg_dist = metric_avg_distance(lca, chunk_types, G)
    # expan_ratio = metric_expansion_ratio(lca, chunk_types, G)
    depth_r = metric_depth_ratio(lca, chunk_types, G, root)
    ic_r = metric_ic_difference(lca, chunk_types, G, cls_inst_count)

    # in_wikc = lca in set(G.nodes())

    return {
        "avg_distance": avg_dist,
        # "expansion_ratio": expan_ratio,
        "depth_ratio": depth_r,
        "ic_difference": ic_r,
        # "in_wikc": in_wikc,
    }


# ---------------------------------------------------------------------------
# Select the best LCA
# ---------------------------------------------------------------------------

def select_lcas(lca_reults, G_clean, root = 'Q35120'):
    toplevel_classes = set(G_clean.successors(root))
    # toplevel_classes = set()
    toplevel_classes.add(root)

    # First pass: collect valid candidates (skip toplevel classes)
    candidates = dict()
    all_covered_types = set()
    for lca, (covered_types, dist) in lca_reults.items():
        if lca in toplevel_classes:
            continue
        candidates[lca] = (set(covered_types), dist)
        all_covered_types.update(covered_types)

    rest_types = all_covered_types.copy()
    lca_selected = dict()

    def _rank_key(stats):
        """Higher is better: (ic rounded 2dp, -avg_distance, -expansion_ratio)."""
        return (
            stats['ic_difference'],
            # -stats['avg_distance'],
            # -stats['expansion_ratio'],
        )

    while rest_types and candidates:
        best_lca = None
        best_stats = None
        best_key = None
        best_covered = None

        for lca, (covered_types, dist) in candidates.items():
            effective_covered = covered_types & rest_types
            if not effective_covered:
                continue
            stats = evaluate_chunk(lca, effective_covered, G_clean, root=root)
            if stats['avg_distance'] <= 2 and stats['ic_difference'] >= -0.1: # 90% of instances
                key = _rank_key(stats)
                if best_key is None or key > best_key:
                    best_key = key
                    best_lca = lca
                    best_stats = stats
                    best_covered = effective_covered

        if best_lca is None:
            break

        _, dist = candidates.pop(best_lca)
        lca_selected[best_lca] = (best_covered, dist, best_stats)
        rest_types -= best_covered

    return lca_selected



# #########################################################
# def _fmt(results):
#     lines = []
#     for lca, (covered, dist) in sorted(results.items(), key=lambda kv: kv[1][1]):
#         cov_str = ", ".join(sorted(covered))
#         label = f" ({qid_to_label[lca]})" if qid_to_label and lca in qid_to_label else ""
#         lines.append(f"    {lca}{label}  dist={dist}  covers=[{cov_str}]")
#     return "\n".join(lines) if lines else "    (none)"


def _constraints_dir() -> Path:
    """Directory containing this file (same as chunk.py: llm/src/constraints)."""
    return Path(__file__).resolve().parent


def _default_rel_constraint_paths() -> tuple[str, str]:
    d = _constraints_dir()
    subj = d / "rel_subject_type_constraints.csv"
    val = d / "rel_value_type_constraints.csv"
    return str(subj), str(val)


def run_metrics_for_type_list(type_list, G, mapping, root="Q35120"):
    """Resolve types, cluster (max_gap), print chunks and LCA/metrics (original layout)."""
    if not type_list:
        print("  (no types)")
        return

    types_qids = [t[0] for t in type_list]
    qid_to_label = {t[0]: t[1] for t in type_list}
    print(f"  {len(types_qids)} types: {chunk.get_labels(types_qids, qid_to_label)}")
    valid_nodes, dropped_nodes = chunk.resolve_nodes(types_qids, G, mapping)
    print("  Non-existing Classes: ", chunk.get_labels(dropped_nodes, qid_to_label))
    types_qids = chunk.remove_redundant(valid_nodes, G)
    print("  Redundant Classes: ", chunk.get_labels(set(valid_nodes) - set(types_qids), qid_to_label))
    print("  Valid Classes: ", chunk.get_labels(types_qids, qid_to_label))

    if not types_qids:
        print("  No valid types after resolve/redundancy removal; skipping clustering.")
        return

    dist = chunk.compute_distance_matrix(types_qids, G)
    chunks, Z, thr = chunk.chunk_by_hierarchical_clustering(
        types_qids, dist, threshold_mode="max_gap"
    )
    chunk.print_chunks("Hierarchical Clustering – max_gap", chunks, qid_to_label)

    print(f"\n{'='*70}")
    print("  LCA & METRICS PER CHUNK")
    print(f"{'='*70}")
    for i, chunk_ in enumerate(chunks):
        lca_results = chunk._reverse_bfs_lcas(chunk_, G)
        print(f"  LCAs for chunk {i+1}: {lca_results}")
        lca_selected = select_lcas(lca_results, G, root=root)
        print(f"    *** Selected LCAs for chunk {i+1}: {lca_selected}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--relation",
        default="P1000",
        help="Wikidata property id (e.g. P1000); subject/object types from constraint CSVs.",
    )
    parser.add_argument(
        "--subject-constraints",
        default=None,
        help="rel_subject_type_constraints.csv (default: same directory as metrics.py)",
    )
    parser.add_argument(
        "--value-constraints",
        default=None,
        help="rel_value_type_constraints.csv (default: same directory as metrics.py)",
    )
    parser.add_argument(
        "--types",
        default=None,
        help="Legacy: single constraints CSV; if set, runs one side only (ignores --relation).",
    )
    args = parser.parse_args()

    G = load_clean_taxonomy("wikc_plus.txt")
    mapping = chunk.load_mapping("mapping.txt")

    if args.types:
        type_list = chunk.load_constraint_types(args.types)
        print("Legacy mode (--types)\n")
        run_metrics_for_type_list(type_list, G, mapping)
    else:
        subj_path = args.subject_constraints or _default_rel_constraint_paths()[0]
        val_path = args.value_constraints or _default_rel_constraint_paths()[1]
        print(f"Property {args.relation}")
        print(f"  subject constraints: {subj_path}")
        print(f"  value constraints:   {val_path}\n")
        subject_types, object_types = chunk.load_relation_constraint_types(
            subj_path, val_path, args.relation
        )
        print("Subject:")
        run_metrics_for_type_list(subject_types, G, mapping)
        print("\nObject:")
        run_metrics_for_type_list(object_types, G, mapping)
