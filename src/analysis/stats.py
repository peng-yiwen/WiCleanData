"""
Single-pass statistics over the wiclean TSV dumps.

Input files (all produced by the wiclean pipeline):
  --facts-file      : TSV with general facts  (subject TAB predicate TAB object TAB .)
  --inst-types-file : TSV with instance-of triples only (same format, predicate == wdt:P31)
  --taxonomy-file   : one 'child,parent' pair per line

Metrics collected:
  - n_entities                : distinct wd: subjects across both TSV files
  - avg_facts_per_entity      : avg triple count per entity, excluding EXCLUDE_PREDICATES
  - total_distinct_predicates
  - n_entities_without_labels : entities that have no rdfs:label triple
  - avg_classes_per_entity    : after expanding wdt:P31 classes with superclasses
  - avg_paths_to_root         : average number of root-reaching paths per entity
"""

import networkx as nx
import os
import csv
from collections import Counter, defaultdict
from tqdm import tqdm
import config


################################################################################
# Paths
ROOT = config.ROOT_QID
DEFAULT_TAXONOMY_FILE = config.TAXONOMY_FILE
DEFAULT_CLS_INST_COUNT_FILE = config.CLS_INST_COUNT_FILE
DEFAULT_SUBJ_CONSTRAINTS_FILE = config.SUBJ_CONSTRAINTS_FILE
DEFAULT_VALUE_CONSTRAINTS_FILE = config.VALUE_CONSTRAINTS_FILE
DEFAULT_FACTS_FILE = config.FACTS_FILE
DEFAULT_INST_TYPES_FILE = config.INST_TYPES_FILE
DEFAULT_OUTPUT_DIR = config.STATS_OUTPUT_DIR


# Predicates excluded when computing avg_facts_per_entity
EXCLUDE_PREDICATES = frozenset([
    "rdfs:label",
    "rdfs:comment",
    "rdf:type",
    "schema:url",
    "owl:sameAs",
    "schema:alternateName",
    "skos:altLabel",
    "schema:description",
    "skos:prefLabel",
])

LABEL_PREDICATE = "rdfs:label"
INSTANCE_OF = "wdt:P31"

################################################################################

def load_constraints(subj_csv: str, value_csv: str) -> dict:
    """Load relation constraints from two CSV files.

    Parameters
    ----------
    subj_csv   : path to rel_constraints_subj_types_clean.csv
                 columns: property, property_label, subject_type, subject_type_label
    value_csv  : path to rel_constraints_value_types_clean.csv
                 columns: property, property_label, value_type, value_type_label

    Returns
    -------
    dict mapping ``wdt:P<id>`` -> RelationConstraint
    """
    constraints = dict()

    # --- subject (domain) types ---
    with open(subj_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            pid = row["property"] # e.g. P10000
            if pid not in constraints:
                constraints[pid] = dict()
            if 'subjectTypes' not in constraints[pid]:
                constraints[pid]['subjectTypes'] = set()
            constraints[pid]['subjectTypes'].add(row["subject_type"])  # e.g. Q5

    # --- object (range) types ---
    with open(value_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            pid = row["property"]
            if pid not in constraints:
                constraints[pid] = dict()
            if 'objectTypes' not in constraints[pid]:
                constraints[pid]['objectTypes'] = set()
            constraints[pid]['objectTypes'].add(row["value_type"])

    return constraints


def load_cls_inst_count(path: str) -> dict:
    cls_inst_count = dict()
    with open(path, 'r') as f:
        for line in f:
            cls, count = line.strip().split('\t')
            cls_inst_count[cls] = int(count)
    return cls_inst_count


def triples_from_tsv(path: str, message: str = ""):
    """
    Yield (subject, predicate, object) tuples from a wiclean TSV dump.

    Each data line has the form:
        subject TAB predicate TAB object TAB .
    Lines starting with '@prefix' or that are empty are skipped.
    """
    with open(path, encoding="utf-8") as fh:
        for line in tqdm(fh, desc=message, unit=" lines"):
            line = line.rstrip("\n")
            if not line or line.startswith("@prefix"):
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            yield parts[0], parts[1], parts[2]


def load_taxonomy_up(path: str) -> dict:
    """
    Load taxonomy file (one 'child,parent' pair per line) into a
    child -> set-of-parents mapping.  IDs are bare QIDs (no 'wd:' prefix).
    """
    taxonomy_up: dict = defaultdict(set)
    print(f"Loading taxonomy from {path} ... ", end="", flush=True)
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",", 1)
            if len(parts) == 2:
                child, parent = parts[0].strip(), parts[1].strip()
                taxonomy_up[child].add(parent)
    taxonomy_up[ROOT] = set()  # add root node
    print("done", flush=True)
    return taxonomy_up


def getSuperClasses(cls, classes, yagoTaxonomyUp, pathsToRoot, counter=0):
    """Adds all superclasses of a class <cls> (including <cls>) to the set <classes>."""
    classes.add(cls)
    if counter > 200:
        print("  Warning: recursion overflow in taxonomy with", cls, "and", classes)
        return False
    if cls == ROOT: # no wd: prefix
        pathsToRoot[0] += 1
    if cls in yagoTaxonomyUp:
        for sc in yagoTaxonomyUp[cls]:
            if not getSuperClasses(sc, classes, yagoTaxonomyUp, pathsToRoot, counter + 1):
                return False
    return True


# ---------------------------------------------------------------------------
#  Core scan (two TSV passes)
# ---------------------------------------------------------------------------

class ScanResult:
    """All counters produced by scanning the two TSV files."""

    def __init__(
        self,
        subject_fact_counts: Counter,
        predicate_counts: Counter,
        subjects_with_label: set,
        entity_direct_classes: dict,
    ) -> None:
        self.subject_fact_counts = subject_fact_counts
        self.predicate_counts = predicate_counts
        self.subjects_with_label = subjects_with_label
        self.entity_direct_classes = entity_direct_classes
        self.totalClassesPerInstance = 0
        self.totalPathsToRoot = 0

    @property
    def n_entities(self) -> int:
        return len(self.subject_fact_counts)

    @property
    def total_distinct_predicates(self) -> int:
        return len(self.predicate_counts)

    @property
    def avg_facts_per_entity(self) -> float:
        if not self.subject_fact_counts:
            return 0.0
        return sum(self.subject_fact_counts.values()) / len(self.subject_fact_counts)

    @property
    def n_entities_without_labels(self) -> int:
        return sum(
            1 for s in self.subject_fact_counts
            if s not in self.subjects_with_label
        )

    def compute_avg_classes_and_paths_per_entity(self, taxonomy_up: dict) -> None:
        """
        For every entity that has ≥1 wdt:P31 class, expand all classes to
        include superclasses, accumulate total class count and root paths.
        """
        print("Expanding classes with superclasses ... ", end="", flush=True)
        for _, direct_classes in self.entity_direct_classes.items():
            superClasses: set = set()
            pathsToRoot = [0]
            for c in direct_classes:
                getSuperClasses(c, superClasses, taxonomy_up, pathsToRoot)
            self.totalClassesPerInstance += len(superClasses)
            self.totalPathsToRoot += pathsToRoot[0]
        # print("done", flush=True)


def scan_tsvs(facts_path: str, inst_types_path: str) -> ScanResult:
    """Two-pass scan: facts TSV then instance-types TSV."""
    subject_fact_counts: Counter = Counter()
    predicate_counts: Counter = Counter()
    subjects_with_label: set = set()
    entity_direct_classes: dict = defaultdict(set)

    # --- pass 1: general facts ---
    for subject, predicate, obj in triples_from_tsv(facts_path, message="Scanning facts TSV"):
        predicate_counts[predicate] += 1
        if subject.startswith("wd:"):
            if predicate == LABEL_PREDICATE:
                subjects_with_label.add(subject)
            if predicate not in EXCLUDE_PREDICATES:
                subject_fact_counts[subject] += 1

    # --- pass 2: instance-type facts ---
    for subject, predicate, obj in triples_from_tsv(inst_types_path, message="Scanning inst-types TSV"):
        predicate_counts[predicate] += 1
        if subject.startswith("wd:") and predicate == INSTANCE_OF and obj.startswith("wd:"):
            qid = subject[3:]
            cls_qid = obj[3:]
            entity_direct_classes[qid].add(cls_qid)
            # ensure the entity appears in subject_fact_counts
            if subject not in subject_fact_counts:
                subject_fact_counts[subject] = 0

    # Entities that appear only in excluded triples still count as entities
    for s in subjects_with_label:
        if s not in subject_fact_counts:
            subject_fact_counts[s] = 0

    return ScanResult(
        subject_fact_counts, predicate_counts,
        subjects_with_label, entity_direct_classes,
    )



def taxonomy_stats(taxonomy_up: dict, cls_inst_count: dict) -> dict:
    root = ROOT
    if not root in taxonomy_up:
        raise ValueError(f"Root node {root} not found in taxonomy")
    # To networkx digraph
    wiki_dag = nx.DiGraph()
    for child, parents in taxonomy_up.items():
        wiki_dag.add_node(child)
        for parent in parents:
            wiki_dag.add_edge(parent, child)
    
    top1level_classes = {node_: taxonomy_up[node_] for node_ in wiki_dag.successors(root)}
    # find n_classes without direct instances
    n_classes_without_direct_instances = 0
    for cls in wiki_dag.nodes():
        if cls not in cls_inst_count:
            n_classes_without_direct_instances += 1

    return {
        "number_of_classes": wiki_dag.number_of_nodes(),
        "number_of_taxonomic_links": wiki_dag.number_of_edges(),
        "number of top1 level classes": len(top1level_classes),
        "max_depth": max(nx.shortest_path_length(wiki_dag, source=root).values()),
        "classes_without_direct_instances": n_classes_without_direct_instances,
        "weakly_connected": nx.is_weakly_connected(wiki_dag),
        "directed_acyclic": nx.is_directed_acyclic_graph(wiki_dag),
        "number_of_roots": len([node for node in wiki_dag.nodes() if not list(wiki_dag.predecessors(node))]),
        "number_of_leaves": len([node for node in wiki_dag.nodes() if wiki_dag.out_degree(node) == 0]),
        "number_of_internal_nodes": len([node for node in wiki_dag.nodes() if wiki_dag.out_degree(node) > 0]),
        "average_in_degree": sum(dict(wiki_dag.in_degree()).values()) / wiki_dag.number_of_nodes(),
    }

def constraint_stats(subj_csv: str, value_csv: str):
    constraints = load_constraints(subj_csv, value_csv)
    # avg number of subj/value constraints per entity
    avg_subj_constraints_per_entity = sum(len(constraints[pid].get('subjectTypes', [])) for pid in constraints) / len(constraints)
    avg_value_constraints_per_entity = sum(len(constraints[pid].get('objectTypes', [])) for pid in constraints) / len(constraints)
    return {
        "avg_subj_constraints_per_entity": avg_subj_constraints_per_entity,
        "avg_value_constraints_per_entity": avg_value_constraints_per_entity,
    }
    

# ---------------------------------------------------------------------------
#  Output helpers
# ---------------------------------------------------------------------------

def write_summary(
    result: ScanResult,
    path: str, # output path
    facts_path: str,
    inst_types_path: str,
    avg_number_of_paths_to_root: float | None = None,
    avg_number_of_classes_per_instance: float | None = None,
) -> None:
    fact_counts = list(result.subject_fact_counts.values())
    n = result.n_entities
    hist = Counter(fact_counts)

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as out:
        # out.write(f"source_facts_file\t\t\t{facts_path}\n")
        # out.write(f"source_inst_types_file\t\t\t{inst_types_path}\n")
        out.write("="*50 + "\n")
        out.write("Summary of the statistics\n")
        out.write("="*50 + "\n")
        out.write(f"n_entities\t\t\t{n:,}\n")
        out.write(f"n_entities_without_labels\t\t{result.n_entities_without_labels:,}\n")
        out.write(f"total_distinct_predicates\t\t{result.total_distinct_predicates:,}\n")
        if n:
            out.write(f"avg_facts_per_entity\t\t\t{result.avg_facts_per_entity:.2f}\n")
            out.write(f"min_facts_per_entity\t\t\t{min(fact_counts):,}\n")
            out.write(f"max_facts_per_entity\t\t\t{max(fact_counts):,}\n")
            median = sorted(fact_counts)[n // 2]
            out.write(f"median_facts_per_entity\t\t\t{median:,}\n")
        if avg_number_of_classes_per_instance is not None:
            out.write(f"avg_classes_per_entity\t\t\t{avg_number_of_classes_per_instance:.2f}\n")
        if avg_number_of_paths_to_root is not None:
            out.write(f"avg_paths_to_root\t\t\t{avg_number_of_paths_to_root:.2f}\n")
        out.write(
            f"\n# note: fact counts exclude predicates: "
            f"{', '.join(sorted(EXCLUDE_PREDICATES))}\n"
        )
        out.write("\n# histogram: facts_per_entity -> num_entities\n")
        for k in sorted(hist):
            out.write(f"{k}\t{hist[k]:,}\n")



if __name__ == "__main__":
    

    for label, path in [("facts", DEFAULT_FACTS_FILE), ("inst-types", DEFAULT_INST_TYPES_FILE)]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{label} file not found: {path}")

    result = scan_tsvs(DEFAULT_FACTS_FILE, DEFAULT_INST_TYPES_FILE)
    cls_inst_count = load_cls_inst_count(DEFAULT_CLS_INST_COUNT_FILE)

    avg_number_of_paths_to_root: float | None = None
    avg_number_of_classes_per_instance: float | None = None
    # avg_number_of_paths_to_root_2: float | None = None
    # avg_number_of_classes_per_instance_2: float | None = None

    # if not args.skip_classes:
    if not os.path.exists(DEFAULT_TAXONOMY_FILE):
        print(
            f"[warning] Taxonomy file not found: {DEFAULT_TAXONOMY_FILE}\n"
            f"          Skipping avg_classes_per_entity. "
        )
    else:
        taxonomy_up = load_taxonomy_up(DEFAULT_TAXONOMY_FILE)
        result.compute_avg_classes_and_paths_per_entity(taxonomy_up)
        avg_number_of_paths_to_root = result.totalPathsToRoot / result.n_entities
        avg_number_of_classes_per_instance = result.totalClassesPerInstance / result.n_entities
        stats_ = taxonomy_stats(taxonomy_up, cls_inst_count)
        # avg_number_of_paths_to_root_2 = result.totalPathsToRoot / len(result.entity_direct_classes)
        # avg_number_of_classes_per_instance_2 = result.totalClassesPerInstance / len(result.entity_direct_classes)

    summary_path = os.path.join(DEFAULT_OUTPUT_DIR, "stats_summary_wiclean.txt")
    write_summary(
        result, summary_path,
        DEFAULT_FACTS_FILE, DEFAULT_INST_TYPES_FILE,
        avg_number_of_paths_to_root,
        avg_number_of_classes_per_instance,
    )
    
    with open(os.path.join(DEFAULT_OUTPUT_DIR, "taxonomy_stats.txt"), "w") as f:
        f.write("="*50 + "\n")
        f.write("Summary of the taxonomy statistics\n")
        f.write("="*50 + "\n")
        for key, value in stats_.items():
            f.write(f"{key}: {value}\n")
    
    with open(os.path.join(DEFAULT_OUTPUT_DIR, "constraint_stats.txt"), "w") as f:
        f.write("="*50 + "\n")
        f.write("Summary of the constraint statistics\n")
        f.write("="*50 + "\n")
        for key, value in constraint_stats(DEFAULT_SUBJ_CONSTRAINTS_FILE, DEFAULT_VALUE_CONSTRAINTS_FILE).items():
            f.write(f"{key}: {value}\n")