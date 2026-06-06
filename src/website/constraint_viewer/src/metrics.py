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
import os
import networkx as nx
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Iterable, Literal

import chunk
import argparse
from pathlib import Path
from config import CLS_INST_COUNT, DATA_DIR, REL_SUBJECT_CSV, REL_VALUE_CSV, WIKC_LABELS, DEFAULT_TAXONOMY_ROOT_ID

# Root node id uses ``Q`` prefix, matching chunk.normalize_qid / load_comma_taxonomy.
TAXONOMY_ROOT = DEFAULT_TAXONOMY_ROOT_ID

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


def get_depth(node, G, root=None):
    r = TAXONOMY_ROOT if root is None else root
    return nx.shortest_path_length(G, r, node)


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

def metric_depth_ratio(lca, chunk_types, G, root=None):
    """
    Depth(LCA) / avg_depth(chunk types).
    G: clean taxonomy graph
    """
    if lca is None:
        raise ValueError("A valid LCA is required")
    r = TAXONOMY_ROOT if root is None else root
    lca_d = get_depth(lca, G, r)
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

    lca_desc = getDescendants(lca, taxonomyDown)
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

def evaluate_chunk(lca, chunk_types, G, root=None):

    # load cls_inst_count from data/
    cls_inst_count = get_cls_instance_count(str(CLS_INST_COUNT))
    # metrics
    avg_dist = metric_avg_distance(lca, chunk_types, G)
    expan_ratio = metric_expansion_ratio(lca, chunk_types, G)
    depth_r = metric_depth_ratio(lca, chunk_types, G, root)
    ic_r = metric_ic_difference(lca, chunk_types, G, cls_inst_count)

    in_wikc = lca in set(G.nodes())

    return {
        "avg_distance": avg_dist,
        "expansion_ratio": expan_ratio,
        "depth_ratio": depth_r,
        "ic_difference": ic_r,
        "in_wikc": in_wikc,
    }


# ---------------------------------------------------------------------------
# Select the best LCA
# ---------------------------------------------------------------------------

def select_lcas(lca_reults, G_clean, root=None):
    r = TAXONOMY_ROOT if root is None else root
    toplevel_classes = set(G_clean.successors(r))
    toplevel_classes.add(r)

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
            round(stats['ic_difference'], 2),
            -stats['avg_distance'],
            -stats['expansion_ratio'],
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
            stats = evaluate_chunk(lca, effective_covered, G_clean, root=r)
            if stats['avg_distance'] <= 2 and stats['ic_difference'] >= -0.1:
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
    """Data directory used by constraint-viewer."""
    return DATA_DIR


def _default_rel_constraint_paths() -> tuple[str, str]:
    return str(REL_SUBJECT_CSV), str(REL_VALUE_CSV)


@lru_cache(maxsize=1)
def load_wikc_labels() -> dict[str, str]:
    """Map Q-id → label from data/wikclabels_2026.txt."""
    path = WIKC_LABELS
    if not path.is_file():
        return {}
    cls2label: dict[str, str] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) < 2:
                continue
            qid, raw = parts[0].strip(), parts[1].strip()
            if len(raw) >= 2 and raw[0] == '"' and raw[-1] == '"':
                lab = raw[1:-1]
            else:
                lab = raw
            cls2label[qid] = lab.replace('""', '"')
    return cls2label


def make_label_resolver(wikc: dict[str, str], csv_map: dict[str, str]):
    """Prefer Wikidata-class label file; fall back to CSV constraint label; else Q-id."""

    def resolve(q: str) -> str:
        w = wikc.get(q)
        if w:
            return w
        c = csv_map.get(q)
        if c and str(c).strip():
            return str(c).strip()
        return q

    return resolve


def _fmt_labeled_list_wikc(qids: list[str], wikc: dict[str, str], csv_map: dict[str, str]) -> str:
    if not qids:
        return "—"
    resolve = make_label_resolver(wikc, csv_map)
    return ", ".join(f"{q} ({resolve(q)})" for q in qids)


def _qid_labels_dict(
    qids: Iterable[str],
    wikc: dict[str, str],
    csv_map: dict[str, str],
) -> dict[str, str]:
    resolve = make_label_resolver(wikc, csv_map)
    return {q: resolve(q) for q in qids}


@dataclass
class SideResult:
    """Per-side (subject or object) pipeline result for unified reporting."""

    side: Literal["subject", "object"]
    original_types: list[tuple[str, str]]
    dropped: list[str]
    redundant: set[str]
    valid_types: list[str]
    selected_lca_details: list[dict[str, Any]]
    mapping_hits: list[tuple[str, str]]
    clean_stats: dict[str, int]


def _label_map(original: list[tuple[str, str]]) -> dict[str, str]:
    return {q: lab for q, lab in original}


def build_clean_stats(
    original_qids: list[str],
    reduced_after_redundancy: list[str],
    selected_lca_details: list[dict[str, Any]],
) -> dict[str, int]:
    """Minimal stats for before/after comparison."""
    return {
        "original_count": len(original_qids),
        "final_count": len(
            final_selected_qids_from_details(
                reduced_after_redundancy, selected_lca_details
            )
        ),
    }


def _fmt_metrics_block(stats: dict[str, Any]) -> str:
    return (
        f"avg_distance={stats['avg_distance']:.4f}, "
        f"expansion_ratio={stats['expansion_ratio']:.4f}, "
        f"depth_ratio={stats['depth_ratio']:.4f}, "
        f"ic_difference={stats['ic_difference']:.4f} "
        # f"in_wikc={stats['in_wikc']}" # in_wikc_by_default
    )


def process_side(
    side: Literal["subject", "object"],
    type_list: list[tuple[str, str]],
    G: nx.DiGraph,
    mapping: dict[str, str],
    root: str | None = None,
) -> SideResult:
    """Resolve → prune redundancy → cluster → LCA selection (multi-node LCAs only)."""
    r = TAXONOMY_ROOT if root is None else root
    if not type_list:
        return SideResult(
            side=side,
            original_types=[],
            dropped=[],
            redundant=set(),
            valid_types=[],
            selected_lca_details=[],
            mapping_hits=[],
            clean_stats=build_clean_stats([], [], []),
        )

    types_qids = [t[0] for t in type_list]
    valid_nodes, dropped_nodes, mapping_hits = chunk.resolve_nodes(
        types_qids, G, mapping, verbose=False, return_mapping_hits=True
    )
    reduced = chunk.remove_redundant(valid_nodes, G)
    redundant_set = set(valid_nodes) - set(reduced)

    selected_details: list[dict[str, Any]] = []
    if reduced:
        dist = chunk.compute_distance_matrix(reduced, G)
        chunks, _, _ = chunk.chunk_by_hierarchical_clustering(
            reduced, dist, threshold_mode="max_gap"
        )
        for chunk_nodes in chunks:
            lca_results = chunk._reverse_bfs_lcas(chunk_nodes, G)
            lca_selected = select_lcas(lca_results, G, root=r)
            for lca, (covered, dist_v, stats) in lca_selected.items():
                if len(covered) < 2:
                    continue
                selected_details.append(
                    {
                        "lca": lca,
                        "covered": set(covered),
                        "dist": dist_v,
                        "metrics": stats,
                    }
                )

    return SideResult(
        side=side,
        original_types=list(type_list),
        dropped=dropped_nodes,
        redundant=redundant_set,
        valid_types=reduced,
        selected_lca_details=selected_details,
        mapping_hits=mapping_hits,
        clean_stats=build_clean_stats(types_qids, reduced, selected_details),
    )


def final_wd_strings_for_side(
    side: SideResult,
    wikc: dict[str, str],
    csv_map: dict[str, str],
) -> list[str]:
    """Build `wd:Q…` strings for the Domain / Range table (final selected types)."""
    resolve = make_label_resolver(wikc, csv_map)
    covered_union: set[str] = set()
    for d in side.selected_lca_details:
        covered_union |= d["covered"]
    out: list[str] = []
    for d in side.selected_lca_details:
        lca = d["lca"]
        lab = resolve(lca)
        out.append(f"wd:{lca} {lab}".strip())
    for q in side.valid_types:
        if q not in covered_union:
            lab = resolve(q)
            out.append(f"wd:{q} {lab}".strip())
    return out


def final_selected_qids_from_details(
    valid_types: list[str],
    selected_lca_details: list[dict[str, Any]],
) -> list[str]:
    """Final selected nodes: selected LCAs + remaining valid types."""
    covered_union: set[str] = set()
    for d in selected_lca_details:
        covered_union |= d["covered"]
    out: list[str] = []
    for d in selected_lca_details:
        out.append(d["lca"])
    out.extend([q for q in valid_types if q not in covered_union])
    return out


def final_mapping_hits_for_side(side: SideResult) -> list[tuple[str, str]]:
    """Keep only mappings whose mapped-to node appears in final selected results."""
    final_set = set(final_selected_qids_from_details(side.valid_types, side.selected_lca_details))
    return [(src, dst) for src, dst in side.mapping_hits if dst in final_set]


def _json_safe_metrics(stats: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in stats.items():
        if isinstance(v, bool):
            out[k] = v
        elif isinstance(v, (int, float)):
            out[k] = v
        else:
            try:
                out[k] = float(v)
            except (TypeError, ValueError):
                out[k] = v
    return out


def side_result_to_json(side: SideResult, wikc: dict[str, str]) -> dict[str, Any]:
    """Serializable snapshot (optional; for debugging or future UI)."""
    csv_map = _label_map(side.original_types)
    return {
        "side": side.side,
        "original_types": [{"qid": q, "label": lab} for q, lab in side.original_types],
        "dropped": list(side.dropped),
        "redundant": sorted(side.redundant),
        "valid_types": list(side.valid_types),
        "mapping_hits": [{"from": a, "to": b} for a, b in side.mapping_hits],
        "final_mapping_hits": [{"from": a, "to": b} for a, b in final_mapping_hits_for_side(side)],
        "clean_stats": dict(side.clean_stats),
        "selected_lca_details": [
            {
                "lca": d["lca"],
                "covered": sorted(d["covered"]),
                "dist": float(d["dist"]) if d["dist"] is not None else None,
                "metrics": _json_safe_metrics(d["metrics"]),
            }
            for d in side.selected_lca_details
        ],
        "final_wd_strings": final_wd_strings_for_side(side, wikc, csv_map),
    }


def format_relation_report(
    relation: str,
    subject: SideResult,
    obj: SideResult,
    wikc: dict[str, str] | None = None,
    relation_label: str | None = None,
) -> str:
    """Same layout as CLI/terminal summary; labels prefer ``wikclabels_2026.txt``."""
    if wikc is None:
        wikc = load_wikc_labels()
    lines: list[str] = []
    bar = "=" * 78
    lines.append(bar)
    rel_title = f"{relation} ({relation_label})" if relation_label else relation
    lines.append(f"Relation {rel_title} — summary")
    lines.append(bar)

    subj_csv = _label_map(subject.original_types)
    obj_csv = _label_map(obj.original_types)

    lines.append("")
    lines.append("(1) Original types (with label)")
    lines.append("  [Subject]")
    lines.append(
        "   "
        + _fmt_labeled_list_wikc([q for q, _ in subject.original_types], wikc, subj_csv)
    )
    lines.append("  [Object]")
    lines.append(
        "   " + _fmt_labeled_list_wikc([q for q, _ in obj.original_types], wikc, obj_csv)
    )

    lines.append("")
    lines.append("(2) Non-existing classes (not in cleaned taxonomy)")
    lines.append("  [Subject]")
    lines.append("   " + _fmt_labeled_list_wikc(subject.dropped, wikc, subj_csv))
    lines.append("  [Object]")
    lines.append("   " + _fmt_labeled_list_wikc(obj.dropped, wikc, obj_csv))

    lines.append("")
    lines.append("(3) Redundant classes (the ancestor of classes already exists)")
    lines.append("  [Subject]")
    lines.append(
        "   " + _fmt_labeled_list_wikc(sorted(subject.redundant), wikc, subj_csv)
    )
    lines.append("  [Object]")
    lines.append(
        "   " + _fmt_labeled_list_wikc(sorted(obj.redundant), wikc, obj_csv)
    )

    lines.append("")
    lines.append("(4) Selected LCAs and metrics")
    for side_res, csv_map in (subject, subj_csv), (obj, obj_csv):
        title = side_res.side.capitalize()
        lines.append("")
        lines.append(f"  [{title}]")
        if not side_res.selected_lca_details:
            lines.append("    —")
            continue
        for i, d in enumerate(side_res.selected_lca_details, 1):
            lca = d["lca"]
            cov = sorted(d["covered"])
            cov_lmap = _qid_labels_dict([lca, *cov], wikc, csv_map)
            lca_lab = cov_lmap[lca]
            subset_str = ", ".join(chunk.get_labels(cov, cov_lmap))
            lines.append(f"    #{i}  LCA {lca} ({lca_lab})")
            lines.append(f"        subset: {subset_str}")
            lines.append(f"        metrics: {_fmt_metrics_block(d['metrics'])}")

    lines.append("")
    lines.append("(5) Final selected types (LCAs + remaining valid types)")
    for side_res, csv_map in (subject, subj_csv), (obj, obj_csv):
        title = side_res.side.capitalize()
        lines.append("")
        lines.append(f"  [{title}]")
        covered_union: set[str] = set()
        for d in side_res.selected_lca_details:
            covered_union |= d["covered"]
        if not side_res.selected_lca_details and not side_res.valid_types:
            lines.append("    —")
            continue
        for d in side_res.selected_lca_details:
            lca = d["lca"]
            rem = sorted(d["covered"])
            row_lmap = _qid_labels_dict([lca, *rem], wikc, csv_map)
            lca_lab = row_lmap[lca]
            cov_labs = ", ".join(chunk.get_labels(rem, row_lmap))
            lines.append(f"    [LCA] {lca} ({lca_lab})  ←  {cov_labs}")
        remainder = [q for q in side_res.valid_types if q not in covered_union]
        for q in remainder:
            lab = make_label_resolver(wikc, csv_map)(q)
            lines.append(f"    [Type] {q} ({lab})")

    lines.append("")
    lines.append("(6) Extra information")
    for side_res, csv_map in (subject, subj_csv), (obj, obj_csv):
        title = side_res.side.capitalize()
        lines.append("")
        lines.append(f"  [{title}]")
        st = side_res.clean_stats
        lines.append(
            "    stats: "
            f"original={st['original_count']}, "
            f"final={st['final_count']}"
        )
        final_hits = final_mapping_hits_for_side(side_res)
        if final_hits:
            lines.append("    found mapping:")
            map_label = make_label_resolver(wikc, csv_map)
            for src, dst in final_hits:
                lines.append(
                    f"      - {src} ({map_label(src)}) -> {dst} ({map_label(dst)})"
                )
        else:
            lines.append("    found mapping: —")

    return "\n".join(lines)


def print_relation_report(
    relation: str,
    subject: SideResult,
    obj: SideResult,
    relation_label: str | None = None,
) -> None:
    print(format_relation_report(relation, subject, obj, None, relation_label))


def compute_relation_analysis(
    relation: str,
    wikc_path: str,
    mapping_path: str,
    subject_csv: str | None = None,
    value_csv: str | None = None,
) -> dict[str, Any]:
    """Run full pipeline; return summary text + final domain/range type strings for the UI."""
    subj_path = subject_csv or _default_rel_constraint_paths()[0]
    val_path = value_csv or _default_rel_constraint_paths()[1]
    if not os.path.isfile(subj_path) or not os.path.isfile(val_path):
        raise FileNotFoundError(f"Missing constraint CSVs: {subj_path} / {val_path}")

    G = chunk.load_comma_taxonomy(wikc_path)
    mapping = chunk.load_comma_mapping(mapping_path)
    subject_types, object_types = chunk.load_relation_constraint_types(
        subj_path, val_path, relation
    )
    subj_res = process_side("subject", subject_types, G, mapping)
    obj_res = process_side("object", object_types, G, mapping)
    wikc = load_wikc_labels()
    subj_csv = _label_map(subj_res.original_types)
    obj_csv = _label_map(obj_res.original_types)
    rel_norm = relation.strip().upper()
    rel_label = chunk.load_property_label(subj_path, val_path, rel_norm)
    return {
        "relation": rel_norm,
        "relation_label": rel_label or "",
        "summary": format_relation_report(rel_norm, subj_res, obj_res, wikc, rel_label),
        "domain_types": final_wd_strings_for_side(subj_res, wikc, subj_csv),
        "range_types": final_wd_strings_for_side(obj_res, wikc, obj_csv),
        "subject": side_result_to_json(subj_res, wikc),
        "object": side_result_to_json(obj_res, wikc),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Metrics for chunked types; taxonomy from {llm}_wikc.txt / {llm}_mapping.txt.",
    )
    parser.add_argument(
        "--relation",
        default="P1000",
        help="Wikidata property id (e.g. P1000).",
    )
    parser.add_argument(
        "--llm",
        default="mistral7b",
        help="Base name for {llm}_wikc.txt and {llm}_mapping.txt (comma-separated Q-ids).",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Directory containing {llm}_wikc.txt and {llm}_mapping.txt (default: directory of metrics.py).",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve() if args.data_dir else None
    wikc_path, map_path = chunk.resolve_llm_taxonomy_paths(args.llm, data_dir)
    if not wikc_path.is_file():
        raise FileNotFoundError(f"Taxonomy not found: {wikc_path}")
    if not map_path.is_file():
        raise FileNotFoundError(f"Mapping not found: {map_path}")

    print(f"Loading taxonomy from {wikc_path} …")
    G = chunk.load_comma_taxonomy(str(wikc_path))
    mapping = chunk.load_comma_mapping(str(map_path))

    default_subj, default_val = _default_rel_constraint_paths()
    if not os.path.isfile(default_subj) or not os.path.isfile(default_val):
        raise FileNotFoundError(
            f"Expected relation constraint CSVs next to metrics.py:\n  {default_subj}\n  {default_val}"
        )
    print(f"Property {args.relation}: types from\n  {default_subj}\n  {default_val}\n")
    subject_types, object_types = chunk.load_relation_constraint_types(
        default_subj, default_val, args.relation
    )
    subj_res = process_side("subject", subject_types, G, mapping)
    obj_res = process_side("object", object_types, G, mapping)
    rel_norm = args.relation.strip().upper()
    rel_lab = chunk.load_property_label(default_subj, default_val, rel_norm)
    print_relation_report(rel_norm, subj_res, obj_res, rel_lab)
