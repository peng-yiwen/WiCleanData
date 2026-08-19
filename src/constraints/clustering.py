"""
This file includes:
1. Chunk type constraints via hierarchical clustering on taxonomy DAG distance.
2. Metrics to evaluate whether an LCA is a good replacement for a chunk of types.
Graph convention: top-down  (parent → child) with single root.

Metrics:
  1. avg_distance    – Average distance from LCA down to each type in the chunk
  2. expansion_ratio – |descendants(LCA)| / |descendants(t) for t in chunk|
  3. depth_ratio     – depth(LCA) / avg_depth(chunk types)
  4. ic_ratio        – IC(LCA) / avg IC(chunk types)
  5. in_wikc         – Whether the LCA appears in the wikc_plus (cleaned Wikidata) taxonomy
"""

from collections import defaultdict, deque
import networkx as nx
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
import math
from collections import defaultdict
import utils


# ---------------------------------------------------------------------------
# Lowest Common Ancestor
# ---------------------------------------------------------------------------
def reverse_bfs_lcas(nodes, G_clean):
    """Step 3: bottom-up BFS that finds LCAs (lowest common ancestors).

    Starting from *nodes*, propagate upward one hop at a time.  An ancestor is
    recorded as an LCA only when its covered set is **not** a subset of any
    already-recorded LCA's covered set — i.e. it represents a genuinely new
    merge of previously disjoint groups.

    Returns:
        dict: {lca_node: [covered_node_set, max_hop_distance(of shortest path from lca to covered node)], ...}
              ordered by ascending distance.
    """
    node_set = set(nodes)

    if len(node_set) <= 1:
        node = next(iter(node_set))
        return {node: [node_set.copy(), 0]}

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

        if len(covered) >= 2: # check if it is an LCA
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


# dag distance via LCA
def dag_distance(u, v, G):
    """
    Shortest distance between u and v through any common ancestor.
    min over all common ancestors a of dist(a→u) + dist(a→v).
    """
    if u == v:
        return 0
    anc_u = nx.ancestors(G, u) | {u} # contain u itself
    anc_v = nx.ancestors(G, v) | {v} # contain v itself
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
# Clean constraint types
# ---------------------------------------------------------------------------

def resolve_nodes(nodes, G_clean, mapping):
    """Step 1: keep only nodes present in G_clean, resolving merges via mapping.

    Returns:
        resolved (list): deduplicated list of nodes that exist in G_clean.
        dropped  (list): nodes absent from both G_clean and mapping.
    """
    resolved, dropped, seen = [], [], set()
    for n in nodes:
        target = n
        while target in mapping:
            # print(f"  Resolving {target} -> {mapping.get(target, target)}")
            target = mapping.get(target, target)
        if target != n:
            print(f"    - Found mapping for {n} -> {target}")
        if target in G_clean:
            if target not in seen:
                seen.add(target)
                resolved.append(target)
        else:
            dropped.append(n)
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
                break

    return lca_selected
