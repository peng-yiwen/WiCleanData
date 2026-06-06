"""
Build summarized relation constraints for *all* properties:

1) resolve valid classes (taxonomy + mapping)
2) hierarchical clustering (max_gap)
3) select LCAs per chunk (metrics.select_lcas)
4) final types = selected LCAs (unique, chunk order) + valid classes not covered by those LCAs

Writes two CSVs with the same columns as rel_subject_type_constraints.csv /
rel_value_type_constraints.csv.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import os
import sys
from pathlib import Path
from typing import Iterable

import networkx as nx

import chunk
from metrics import load_clean_taxonomy, select_lcas


# ---------------------------------------------------------------------------
# Customizable default paths (edit here, or override via CLI args)
# ---------------------------------------------------------------------------

DEFAULT_ROOT_QID = "Q35120"

# This provides labels for *all* QIDs (including newly selected LCAs).
DEFAULT_LABELS_TSV = (
    Path(__file__).resolve().parents[2]
    / "quick_check"
    / "data"
    / "clean"
    / "wiki_2026_filtered_labels_v3.tsv"
)


def _constraints_dir() -> Path:
    return Path(__file__).resolve().parent


def _default_rel_constraint_paths() -> tuple[str, str]:
    d = _constraints_dir()
    return str(d / "rel_subject_type_constraints.csv"), str(d / "rel_value_type_constraints.csv")


@contextlib.contextmanager
def _chdir(path: Path):
    prev = Path.cwd()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(prev)


def collect_properties_and_labels(
    subject_csv: str,
    value_csv: str,
) -> tuple[list[str], dict[str, str]]:
    """
    Returns:
        sorted property ids (union of both files)
        property_id -> property_label (first non-empty seen)
    """
    properties: set[str] = set()
    prop_labels: dict[str, str] = {}

    def ingest_row(prop: str, prop_label: str) -> None:
        properties.add(prop)
        if prop not in prop_labels and prop_label:
            prop_labels[prop] = prop_label

    with open(subject_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            p = chunk._normalize_property_id(row.get("property", ""))
            if not p:
                continue
            ingest_row(p, (row.get("property_label") or "").strip())

    with open(value_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            p = chunk._normalize_property_id(row.get("property", ""))
            if not p:
                continue
            ingest_row(p, (row.get("property_label") or "").strip())

    return sorted(properties), prop_labels


def load_qid_labels_tsv(path: str) -> dict[str, str]:
    """
    Load a TSV like: QID <tab> label
    Header row is allowed (e.g. 'QID\\tlabel').
    """
    out: dict[str, str] = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if not row:
                continue
            if row[0].strip().upper() == "QID":
                continue
            if len(row) < 2:
                continue
            qid = row[0].strip()
            label = row[1].strip().strip('"')
            if qid and qid not in out and label:
                out[qid] = label
    return out


def load_type_labels_from_constraint_csvs(subject_csv: str, value_csv: str) -> dict[str, str]:
    """Fallback labels from the original constraint CSVs (first occurrence wins)."""
    out: dict[str, str] = {}
    with open(subject_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            qid = (row.get("subject_type") or "").strip()
            lab = (row.get("subject_type_label") or "").strip()
            if qid and qid not in out and lab:
                out[qid] = lab
    with open(value_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            qid = (row.get("value_type") or "").strip()
            lab = (row.get("value_type_label") or "").strip()
            if qid and qid not in out and lab:
                out[qid] = lab
    return out

def compute_selected_lcas_for_side(
    type_list: list[tuple[str, str]],
    G: nx.DiGraph,
    mapping: dict[str, str],
    root: str = "Q35120",
) -> list[dict]:
    """Return algorithm-selected LCAs for one side (subject or object).
    Each entry:
        lca (str): QID in the clean taxonomy (wikc_plus), not necessarily in
            the original constraint type list.
        covered_types (list[str]): original constraint types summarized by this LCA.
        dist (int): max hop distance within the chunk (from reverse BFS).

        stats (dict): metric scores from select_lcas (avg_distance, depth_ratio, ic_difference).
    """
    if not type_list:
        return []
    types_qids = [t[0] for t in type_list]
    valid_nodes, _ = chunk.resolve_nodes(types_qids, G, mapping)
    valid_classes = chunk.remove_redundant(valid_nodes, G)
    if not valid_classes:
        return []
    dist = chunk.compute_distance_matrix(valid_classes, G)
    chunks, _, _ = chunk.chunk_by_hierarchical_clustering(
        valid_classes,
        dist,
        threshold_mode="max_gap",
    )
    records: list[dict] = []
    seen_lcas: set[str] = set()
    for chunk_nodes in chunks:
        lca_results = chunk._reverse_bfs_lcas(chunk_nodes, G)
        lca_selected = select_lcas(lca_results, G, root=root)
        for lca, (covered_types, dist_to_chunk, stats) in lca_selected.items():
            if lca in seen_lcas:
                continue
            seen_lcas.add(lca)
            records.append(
                {
                    "lca": lca,
                    "covered_types": sorted(covered_types),
                    "dist": dist_to_chunk,
                    "stats": stats,
                }
            )
    return records



def compute_final_type_qids_for_side(
    type_list: list[tuple[str, str]],
    G: nx.DiGraph,
    mapping: dict[str, str],
    root: str = "Q35120",
) -> list[str]:
    """LCA summarization for one side; LCAs deduped in chunk iteration order."""
    if not type_list:
        return []

    types_qids = [t[0] for t in type_list]
    valid_nodes, _ = chunk.resolve_nodes(types_qids, G, mapping)
    valid_classes = chunk.remove_redundant(valid_nodes, G)
    if not valid_classes:
        return []

    dist = chunk.compute_distance_matrix(valid_classes, G)
    chunks, _, _ = chunk.chunk_by_hierarchical_clustering(
        valid_classes,
        dist,
        threshold_mode="max_gap",
    )

    ordered_lcas: dict[str, None] = {}
    covered_by_lcas: set[str] = set()
    for chunk_nodes in chunks:
        lca_results = chunk._reverse_bfs_lcas(chunk_nodes, G)
        lca_selected = select_lcas(lca_results, G, root=root)
        for lca in lca_selected:
            ordered_lcas.setdefault(lca, None)
        for _lca, (covered_types, _dist, _stats) in lca_selected.items():
            covered_by_lcas.update(covered_types)

    remaining = [t for t in valid_classes if t not in covered_by_lcas]
    return list(ordered_lcas.keys()) + remaining


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
    labels_tsv: str,
    taxonomy_path: str,
    mapping_path: str,
    root: str,
    quiet: bool,
    limit: int | None,
) -> None:
    subject_in = str(Path(subject_in).resolve())
    value_in = str(Path(value_in).resolve())
    subject_out = str(Path(subject_out).resolve())
    value_out = str(Path(value_out).resolve())
    labels_tsv = str(Path(labels_tsv).resolve())
    taxonomy_path = str(Path(taxonomy_path).resolve())
    mapping_path = str(Path(mapping_path).resolve())

    props, prop_labels = collect_properties_and_labels(subject_in, value_in)
    if limit is not None:
        props = props[:limit]

    constraints_root = _constraints_dir()
    type_labels = load_qid_labels_tsv(labels_tsv)
    # Fallback to original CSV labels if TSV misses anything
    type_labels |= load_type_labels_from_constraint_csvs(subject_in, value_in)

    print(f"Loading taxonomy {taxonomy_path!r} and mapping {mapping_path!r} ...", file=sys.stderr)
    with _chdir(constraints_root):
        G = load_clean_taxonomy(taxonomy_path)
        mapping = chunk.load_mapping(mapping_path)

        subj_rows: list[tuple[str, str, str, str]] = []
        val_rows: list[tuple[str, str, str, str]] = []

        for i, prop in enumerate(props):
            if not quiet and (i == 0 or (i + 1) % 50 == 0 or i + 1 == len(props)):
                print(f"  [{i + 1}/{len(props)}] {prop}", file=sys.stderr)

            plab = prop_labels.get(prop, "")

            ctx = contextlib.redirect_stdout(io.StringIO()) if quiet else contextlib.nullcontext()
            with ctx:
                subj_types, obj_types = chunk.load_relation_constraint_types(
                    subject_in, value_in, prop
                )
                subj_final = compute_final_type_qids_for_side(subj_types, G, mapping, root=root)
                obj_final = compute_final_type_qids_for_side(obj_types, G, mapping, root=root)

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize all relation type constraints.")
    d = _constraints_dir()
    subj_def, val_def = _default_rel_constraint_paths()
    parser.add_argument("--subject-constraints", default=subj_def, help="Input subject CSV")
    parser.add_argument("--value-constraints", default=val_def, help="Input value CSV")
    parser.add_argument(
        "--out-subject",
        default=str(d / "rel_subject_type_constraints_final.csv"),
        help="Output subject constraints CSV",
    )
    parser.add_argument(
        "--out-value",
        default=str(d / "rel_value_type_constraints_final.csv"),
        help="Output value constraints CSV",
    )
    parser.add_argument(
        "--taxonomy",
        default=str(d / "wikc_plus.txt"),
        help="Clean taxonomy (tab-separated, wd:Q child / parent)",
    )
    parser.add_argument(
        "--mapping",
        default=str(d / "mapping.txt"),
        help="QID merge mapping (tab-separated)",
    )
    parser.add_argument(
        "--labels-tsv",
        default=str(DEFAULT_LABELS_TSV),
        help="QID->label TSV (QID<tab>label), used to fill labels for LCAs and missing types",
    )
    parser.add_argument("--root", default=DEFAULT_ROOT_QID, help="Taxonomy root QID (no wd: prefix)")
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress chunk/mapping stdout noise during batch run",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N properties (sorted), for smoke tests",
    )
    args = parser.parse_args()

    run_all(
        args.subject_constraints,
        args.value_constraints,
        args.out_subject,
        args.out_value,
        args.labels_tsv,
        args.taxonomy,
        args.mapping,
        args.root,
        args.quiet,
        args.limit,
    )


if __name__ == "__main__":
    main()
