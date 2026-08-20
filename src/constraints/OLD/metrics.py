"""
Metrics to evaluate whether an LCA is a good replacement for a chunk of types.

Metrics:
  1. avg_distance    – Average distance from LCA down to each type in the chunk
  2. expansion_ratio – |descendants(LCA)| / |descendants(t) for t in chunk|
  3. depth_ratio     – depth(LCA) / avg_depth(chunk types)
  4. ic_ratio        – IC(LCA) / avg IC(chunk types)
  5. in_wikc         – Whether the LCA appears in the wikc_plus (cleaned Wikidata) taxonomy
"""

import math
import csv
import networkx as nx
from collections import defaultdict
import utils
# from config import CLS_INSTANCE_COUNT_PATH


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
    lca_d = utils.get_depth(lca, G, root)
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

    lca_desc = utils.getDescendants(lca, taxonomyDown) # including lca itself
    c_lca = sum(cls_inst_count.get(d, 0) for d in lca_desc)

    chunk_desc = set()
    for t in chunk_types:
        chunk_desc.update(utils.getDescendants(t, taxonomyDown)) # including t itself

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

def evaluate_chunk(lca, chunk_types, G, cls_inst_count, root='Q35120'):
    """Evaluate all metrics for one chunk types"""
    # all metrics
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

def _rank_key(stats):
    """Higher is better: (ic rounded 2dp, -avg_distance, -expansion_ratio)."""
    return (
        stats['ic_difference'],
        -stats['avg_distance'],
        # -stats['expansion_ratio'],
    )

def select_lcas(lca_results, G_clean, thresholds, cls_inst_count, root = 'Q35120'):
    """Select the best LCA for each chunk"""
    
    # get all toplevel classes
    toplevel_classes = set(G_clean.successors(root))
    toplevel_classes.add(root)

    # first pass: collect valid candidates 
    # TBC: skip toplevel classes or not? skip
    candidates = dict()
    all_covered_types = set()
    for lca, (covered_types, distance) in lca_results.items():
        if lca in toplevel_classes:
            continue
        candidates[lca] = (set(covered_types), distance)
        all_covered_types.update(covered_types)

    # second pass: select the best LCA
    rest_types = all_covered_types.copy()
    lca_selected = dict()
    while rest_types and candidates:
        best_lca = None
        best_stats = None
        best_key = None
        best_covered = None
        # each time, select the best LCA (best coverage or best metrics) from all rest candidates
        for lca, (covered_types, dist) in candidates.items():
            effective_covered = covered_types & rest_types
            # if not effective_covered:
            #     continue
            if len(effective_covered) <= 1: # skip if the LCA covers only one rest type
                continue
            stats = evaluate_chunk(lca, effective_covered, G_clean, cls_inst_count, root=root)
            if stats['ic_difference'] >= thresholds['ic_difference'] and stats['avg_distance'] <= thresholds['avg_distance']:
            # if stats['avg_distance'] <= 2 and stats['ic_difference'] >= -0.1: # 90% of instances
                key = _rank_key(stats)
                # CHANGED: 08-19-2026: select the best LCA (best coverage or best metrics)
                if best_key is None:
                    best_key = key
                    best_lca = lca
                    best_stats = stats
                    best_covered = effective_covered
                    continue
                # best coverage
                if len(best_covered) < len(effective_covered):
                    best_key = key
                    best_lca = lca
                    best_stats = stats
                    best_covered = effective_covered
                    continue
                # best metrics
                if len(best_covered) == len(effective_covered) and key > best_key:
                    best_key = key
                    best_lca = lca
                    best_stats = stats
                    best_covered = effective_covered
                    continue

        if best_lca is None:
            break

        _, dist = candidates.pop(best_lca)
        lca_selected[best_lca] = (best_covered, dist, best_stats)
        rest_types -= best_covered
    
    # CHANGED: 08-19-2026: filter lcas that are already covered by other lcas
    lcas = list(lca_selected.keys())
    for i in range(len(lcas)):
        if lcas[i] not in lca_selected:
            continue
        for j in range(i+1, len(lcas)):
            if lcas[j] not in lca_selected:
                continue
            if nx.has_path(G_clean, lcas[i], lcas[j]):
                lca_selected.pop(lcas[j])
            if nx.has_path(G_clean, lcas[j], lcas[i]):
                lca_selected.pop(lcas[i])

    return lca_selected

