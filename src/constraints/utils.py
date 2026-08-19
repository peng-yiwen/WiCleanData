import csv
import networkx as nx
from collections import defaultdict


def load_labels(file_loc):
    labels = dict()
    with open(file_loc, "r") as f:
        for line in f:
            terms = line.strip().split('\t')
            if len(terms) != 2:
                continue
            qid, label = terms
            labels[qid] = label[1:-1] # remove quotes
    return labels


def load_clean_taxonomy(file_loc):
    cleanWikiTaxonDown = defaultdict(set)
    with open(file_loc, 'r') as clean:
        for line in clean:
            terms = line.strip().split(',')
            if len(terms) != 2:
                continue
            child, parent = terms
            cleanWikiTaxonDown[parent].add(child) # no wd: prefix
    return nx.DiGraph(cleanWikiTaxonDown)


def load_mapping(path: str):
    mapping = dict()
    with open(path) as f:
        for line in f:
            terms = line.strip().split(',')
            if len(terms) != 2:
                continue
            qid, parent = terms
            mapping[qid] = parent # no wd: prefix
    return mapping


def load_cls_instance_count(file_loc):
    cls_instance_count = dict()
    with open(file_loc, "r") as f:
        for line in f:
            terms = line.strip().split('\t')
            if len(terms) != 2:
                continue
            cls, count = terms
            cls_instance_count[cls] = int(count) # no wd: prefix
    return cls_instance_count


def _normalize_property_id(property_id: str) -> str:
    pid = property_id.strip()
    if pid.startswith("http://www.wikidata.org/entity/"):
        pid = pid[len("http://www.wikidata.org/entity/"):]
    return pid.upper()


def load_relation_constraint_types(
    subject_csv_path: str,
    value_csv_path: str,
    property_id: str,
):
    """Load subject-side and object-side types for a Wikidata property.
    Rows are matched on the ``property`` column against *subject_csv* and *value_csv*.

    Returns:
        (subject_types, object_types): each a list of (qid, label), order-preserving
        with first occurrence kept when the same QID appears on multiple rows.
    """
    prop = _normalize_property_id(property_id)

    subjects: list[tuple[str, str]] = []
    with open(subject_csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if _normalize_property_id(row.get("property", "")) != prop:
                continue
            subjects.append((row["class"], row["classLabel"]))

    objects: list[tuple[str, str]] = []
    with open(value_csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if _normalize_property_id(row.get("property", "")) != prop:
                continue
            objects.append((row["class"], row["classLabel"]))

    return set(subjects), set(objects)


def load_all_contraints_and_labels(
    subject_csv_path: str,
    value_csv_path: str):
    """Load all constraints and labels for a Wikidata property."""
    props, prop_labels = [], {}
    constraint_type_labels = {}
    prop2constraint_types = defaultdict(dict)

    with open(subject_csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            prop = _normalize_property_id(row.get("property", ""))
            if prop not in props:
                props.append(prop)
                prop_labels[prop] = row.get("propertyLabel", "")
            constraint_type_labels[row.get("class")] = row.get("classLabel", "")
            if 'subject' not in prop2constraint_types[prop]:
                prop2constraint_types[prop]['subject'] = set()
            prop2constraint_types[prop]['subject'].add(row.get("class", ""))

    with open(value_csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            prop = _normalize_property_id(row.get("property", ""))
            if prop not in props:
                props.append(prop)
                prop_labels[prop] = row.get("propertyLabel", "")
            constraint_type_labels[row.get("class")] = row.get("classLabel", "")
            if 'object' not in prop2constraint_types[prop]:
                prop2constraint_types[prop]['object'] = set()
            prop2constraint_types[prop]['object'].add(row.get("class", ""))
    return props, prop_labels, constraint_type_labels, prop2constraint_types



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


def get_depth(node, G, root='Q35120'):
    if root is None:
        raise ValueError("Root is required")
    return nx.shortest_path_length(G, root, node)