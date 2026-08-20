"""
Build summarized relation constraints for *all* properties:

1) resolve valid classes (taxonomy + mapping)
2) hierarchical clustering (max_gap)
3) select LCAs per chunk (metrics.select_lcas)
4) final types = selected LCAs (unique, chunk order) + valid classes not covered by those LCAs
5) write to CSV
"""

import contextlib
import csv
import io
import sys
from pathlib import Path
from typing import Iterable
import networkx as nx
import clustering
import config
import utils


def clean_constraint_types_for_side(
    types: list[str],
    G: nx.DiGraph,
    mapping: dict[str, str],
    cls_inst_count: dict[str, int],
    thresholds: dict[str, float],
    root: str = "Q35120",
) -> list[str]:
    """Clean constraint types for one side."""
    if not types:
        return []

    # step 1: pre-processing: resolve nodes and remove redundant nodes
    valid_nodes, _ = clustering.resolve_nodes(types, G, mapping)
    valid_classes = clustering.remove_redundant(valid_nodes, G)
    if not valid_classes:
        return []
    
    # step 2: hierarchical clustering
    distance = clustering.compute_distance_matrix(valid_classes, G)
    chunks, _, _ = clustering.chunk_by_hierarchical_clustering(
        valid_classes,
        distance,
        threshold_mode="max_gap",
    )
    
    # step 3: select LCAs and keep the remaining types
    ordered_lcas: dict[str, None] = {}
    covered_by_lcas: set[str] = set()
    for chunk_nodes in chunks:
        lca_results = clustering.reverse_bfs_lcas(chunk_nodes, G)
        lca_selected = clustering.select_lcas(lca_results, G, thresholds, cls_inst_count, root=root)
        for lca in lca_selected:
            ordered_lcas.setdefault(lca, None)
        for _lca, (covered_types, _dist, _stats) in lca_selected.items():
            covered_by_lcas.update(covered_types)

    remaining = [t for t in valid_classes if t not in covered_by_lcas]
    final_types = list(ordered_lcas.keys()) + remaining
    return set(final_types)



def _write_subject_csv(
    path: str,
    rows: Iterable[tuple[str, str, str, str]],
) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["property", "property_label", "subject_type", "subject_type_label"])
        w.writerows(rows)


def _write_value_csv(
    path: str,
    rows: Iterable[tuple[str, str, str, str]],
) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["property", "property_label", "value_type", "value_type_label"])
        w.writerows(rows)


def run_all(
    subject_in: str,
    value_in: str,
    subject_out: str,
    value_out: str,
    wiki_labels: str,
    taxonomy_path: str,
    mapping_path: str,
    cls_inst_count_path: str,
    thresholds: dict[str, float],
    root: str,
    quiet: bool,
    # limit: int | None,
) -> None:
    subject_in = str(Path(subject_in).resolve())
    value_in = str(Path(value_in).resolve())
    subject_out = str(Path(subject_out).resolve())
    value_out = str(Path(value_out).resolve())
    wiki_labels = str(Path(wiki_labels).resolve())
    taxonomy_path = str(Path(taxonomy_path).resolve())
    mapping_path = str(Path(mapping_path).resolve())

    wicleanTaxonomy = utils.load_clean_taxonomy(taxonomy_path)
    mapping = utils.load_mapping(mapping_path)
    type_labels = utils.load_labels(wiki_labels)
    props, prop_labels, constraint_type_labels, prop2constraint_types = utils.load_all_contraints_and_labels(subject_in, value_in)
    cls_inst_count = utils.load_cls_instance_count(cls_inst_count_path)

    # update type_labels dict with constraint_type_labels dict
    for qid, label in constraint_type_labels.items():
        if qid not in type_labels:
            type_labels[qid] = label

    # clustering and finding lcas
    subj_rows: list[tuple[str, str, str, str]] = []
    val_rows: list[tuple[str, str, str, str]] = []

    for i, prop in enumerate(props):
        if (i == 0 or (i + 1) % 50 == 0 or i + 1 == len(props)):
            print(f"  [{i + 1}/{len(props)}] {prop}", file=sys.stderr)

        plab = prop_labels.get(prop, "")

        ctx = contextlib.redirect_stdout(io.StringIO()) if quiet else contextlib.nullcontext()
        with ctx:
            subj_types = list(prop2constraint_types[prop].get('subject', set()))
            obj_types = list(prop2constraint_types[prop].get('object', set()))
            subj_final = clean_constraint_types_for_side(subj_types, wicleanTaxonomy, mapping, cls_inst_count, thresholds, root=root)
            obj_final = clean_constraint_types_for_side(obj_types, wicleanTaxonomy, mapping, cls_inst_count, thresholds, root=root)

        for qid in subj_final:
            subj_rows.append((prop, plab, qid, type_labels.get(qid, "")))
        for qid in obj_final:
            val_rows.append((prop, plab, qid, type_labels.get(qid, "")))

    _write_subject_csv(subject_out, subj_rows)
    _write_value_csv(value_out, val_rows)
    print(
        f"Wrote {len(subj_rows)} subject rows -> {subject_out}\n"
        f"Wrote {len(val_rows)} value rows -> {value_out}",
        file=sys.stderr,
    )


if __name__ == "__main__":

    # file paths
    SUBJECT_CONSTRAINTS_CSV = config.SUBJECT_CONSTRAINTS_CSV
    VALUE_CONSTRAINTS_CSV = config.VALUE_CONSTRAINTS_CSV
    SUBJECT_CONSTRAINTS_CLEAN_CSV = config.SUBJECT_CONSTRAINTS_OUT_CSV
    VALUE_CONSTRAINTS_CLEAN_CSV = config.VALUE_CONSTRAINTS_OUT_CSV
    WIKI_LABELS = config.TAXONOMY_LABELS_FILE
    TAXONOMY_PATH = config.WICLEAN_TAXONOMY_FILE
    MAPPING_PATH = config.WICLEAN_MAPPING_FILE
    CLS_INSTANCE_COUNT_PATH = config.CLS_INST_COUNT_FILE
    ROOT_QID = config.ROOT_QID
    thresholds = config.THRESHOLDS

    run_all(
        SUBJECT_CONSTRAINTS_CSV,
        VALUE_CONSTRAINTS_CSV,
        SUBJECT_CONSTRAINTS_CLEAN_CSV,
        VALUE_CONSTRAINTS_CLEAN_CSV,
        WIKI_LABELS,
        TAXONOMY_PATH,
        MAPPING_PATH,
        CLS_INSTANCE_COUNT_PATH,
        thresholds,
        ROOT_QID,
        quiet=True,
    )
