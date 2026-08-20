import csv
import TsvUtils
import Prefixes
import NtUtils
from collections import defaultdict
import config
import os

# Paths (defined in config.py)
SUBJ_CSV           = config.TYPECHECK_SUBJ_CSV
VALUE_CSV          = config.TYPECHECK_VALUE_CSV
INST_TYPE_PATH     = config.TYPECHECK_INST_TYPE_PATH
FACTS_PATH         = config.TYPECHECK_FACTS_PATH
TAXONOMY_PATH      = config.TAXONOMY_FILE
OUTPUT_FOLDER      = config.TYPECHECK_OUTPUT_FOLDER
META_MESSAGES_FILE = config.TYPECHECK_META_MESSAGES_FILE
FACTS_FILE         = config.TYPECHECK_FACTS_FILE

##########################################################################
#             Constraints Class
##########################################################################

class RelationConstraint:
    """Represents a relation with its domain (subjectTypes) and range (objectTypes) constraints."""

    def __init__(self, identifier: str, label: str = ""):
        self.identifier = identifier
        self.rellabel = label
        self.subjectTypes: set = set()   # allowed domain types (classes the subject must belong to)
        self.objectTypes: set = set()    # allowed range  types (classes the object  must belong to)
        self.subjectTypesCount: dict = defaultdict(int) # to calculate the statisfied subjects facts count for each subject type
        self.objectTypesCount: dict = defaultdict(int) # to calculate the statisfied objects facts count for each object type

    # ------------------------------------------------------------------
    # Constraint checking
    # ------------------------------------------------------------------

    def checkSubject(self, entity_types: set) -> bool:
        """Returns True if *entity_types* satisfies the domain constraint.
        An empty subjectTypes set means no constraint (always passes)."""
        if not self.subjectTypes:
            return True
        return bool(entity_types & self.subjectTypes)

    def checkObject(self, entity_types: set) -> bool:
        """Returns True if *entity_types* satisfies the range constraint.
        An empty objectTypes set means no constraint (always passes)."""
        if not self.objectTypes:
            return True
        return bool(entity_types & self.objectTypes)

    def check(self, subject_types: set, object_types: set) -> bool:
        """Returns True only when both domain and range constraints are satisfied."""
        return self.checkSubject(subject_types) and self.checkObject(object_types)

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __eq__(self, other):
        try:
            return other and other.identifier == self.identifier
        except Exception:
            return False

    def __hash__(self):
        return hash(self.identifier)

    def __lt__(self, other):
        try:
            return self.identifier < other.identifier
        except Exception:
            return False

    def __repr__(self):
        return (
            f"RelationConstraint({self.identifier!r}, label={self.rellabel!r}, "
            f"subjectTypes={self.subjectTypes}, objectTypes={self.objectTypes})"
        )

    def __str__(self):
        return self.identifier


##########################################################################
#             Loading constraints from CSV files
##########################################################################

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
    constraints: dict[str, RelationConstraint] = {}

    # --- subject (domain) types ---
    with open(subj_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            pid = "wdt:" + row["property"] # e.g. wdt:P10000
            if pid not in constraints:
                constraints[pid] = RelationConstraint(pid, label=row["property_label"])
            constraints[pid].subjectTypes.add("wd:" + row["subject_type"])  # e.g. wd:Q5

    # --- object (range) types ---
    with open(value_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            pid = "wdt:" + row["property"]
            if pid not in constraints:
                constraints[pid] = RelationConstraint(pid, label=row["property_label"])
            constraints[pid].objectTypes.add("wd:" + row["value_type"])

    return constraints


##########################################################################
#             Useful functions
##########################################################################

def isSubClassOfAny_(c, superclasses, seenClasses, cleanTaxonomyUp):
    """ True if this class is a subclass of any of the given superclasses, avoiding loops"""
    if c in seenClasses: # avoid loops
        return False
    if c in superclasses:
        return True
    if c not in cleanTaxonomyUp:
        return False
    seenClasses.add(c)
    for superclass in cleanTaxonomyUp[c]:
        if isSubClassOfAny_(superclass, superclasses, seenClasses, cleanTaxonomyUp):
            return True
    seenClasses.discard(c)
    return False

def isSubClassOfAny(c, superclasses, wicleanTaxonomyUp):
    """ True if this class is a subclass of any of the given superclasses"""
    # Can't use default argument as this is instantiated only once
    return isSubClassOfAny_(c, superclasses, set(), wicleanTaxonomyUp) 
    

##########################################################################
#             Constraint Checks
##########################################################################

def isValidDomain(triple, subjectCurTypes, subjectConstraintClasses, cleanTaxonomyUp, writerMetaMessages):
    s, p, o = triple
    # no constraints means always valid
    if len(subjectConstraintClasses) == 0:
        return True
    
    # if type constraints exist but no curtypes for subject, is not valid
    # no curtypes reason: invalid instances (no label), fail to retype ...etc
    if len(subjectCurTypes) == 0:
        writerMetaMessages.write("<<",s, p, o, ">>", Prefixes.wicReason, "no_types_for_subject", ".")
        return False

    # domain types check
    for cls_ in subjectCurTypes:
        if isSubClassOfAny(cls_, subjectConstraintClasses, cleanTaxonomyUp):
            return True
    writerMetaMessages.write("<<",s, p, o, ">>", Prefixes.wicReason, "domain_constraint_not_satisfied", ".")
    return False



def isValidRange(triple, objectCurTypes, objectConstraintClasses, cleanTaxonomyUp, writerMetaMessages):
    s, p, o = triple
    # no constraints means always valid
    if len(objectConstraintClasses) == 0:
        return True
    
    if len(objectCurTypes) == 0:
        writerMetaMessages.write("<<",s, p, o, ">>", Prefixes.wicReason, "no_types_for_object", ".")
        return False
    
    # range types check
    for cls_ in objectCurTypes:
        if isSubClassOfAny(cls_, objectConstraintClasses, cleanTaxonomyUp):
            return True
    
    writerMetaMessages.write("<<",s, p, o, ">>", Prefixes.wicReason, "range_constraint_not_satisfied", ".")
    return False



# test functions
def isValidDomain_noType(triple, subjectCurTypes, subjectConstraintClasses, cleanTaxonomyUp):
    s, p, o = triple
    # no constraints means always valid
    if len(subjectConstraintClasses) == 0:
        return True
    
    # if type constraints exist but no curtypes for subject, is not valid
    # no curtypes reason: invalid instances (no label), fail to retype ...etc
    if len(subjectCurTypes) == 0:
        # writerMetaMessages.write("<<",s, p, o, ">>", Prefixes.wicReason, "no_types_for_subject", ".")
        return False

    # # domain types check
    # for cls_ in subjectCurTypes:
    #     if isSubClassOfAny(cls_, subjectConstraintClasses, cleanTaxonomyUp):
    #         return True
    # writerMetaMessages.write("<<",s, p, o, ">>", Prefixes.wicReason, "domain_constraint_not_satisfied", ".")
    return True



def isValidRange_noType(triple, objectCurTypes, objectConstraintClasses, cleanTaxonomyUp):
    s, p, o = triple
    # no constraints means always valid
    if len(objectConstraintClasses) == 0:
        return True
    
    if len(objectCurTypes) == 0:
        # writerMetaMessages.write("<<",s, p, o, ">>", Prefixes.wicReason, "no_types_for_object", ".")
        return False
    
    # # range types check
    # for cls_ in objectCurTypes:
    #     if isSubClassOfAny(cls_, objectConstraintClasses, cleanTaxonomyUp):
    #         return True
    
    # writerMetaMessages.write("<<",s, p, o, ">>", Prefixes.wicReason, "range_constraint_not_satisfied", ".")
    return True


def isValidDomain_violate(triple, subjectCurTypes, subjectConstraintClasses, cleanTaxonomyUp):
    s, p, o = triple
    # no constraints means always valid
    if len(subjectConstraintClasses) == 0:
        return True
    
    # if type constraints exist but no curtypes for subject, is not valid
    # no curtypes reason: invalid instances (no label), fail to retype ...etc
    # if len(subjectCurTypes) == 0:
        # writerMetaMessages.write("<<",s, p, o, ">>", Prefixes.wicReason, "no_types_for_subject", ".")
        # return False

    if len(subjectCurTypes) > 0:
        # domain types check
        for cls_ in subjectCurTypes:
            if isSubClassOfAny(cls_, subjectConstraintClasses, cleanTaxonomyUp):
                return True
        return False
        # writerMetaMessages.write("<<",s, p, o, ">>", Prefixes.wicReason, "domain_constraint_not_satisfied", ".")
    return True


def isValidRange_violate(triple, objectCurTypes, objectConstraintClasses, cleanTaxonomyUp):
    s, p, o = triple
    # no constraints means always valid
    if len(objectConstraintClasses) == 0:
        return True
    
    # if len(objectCurTypes) == 0:
    #     # writerMetaMessages.write("<<",s, p, o, ">>", Prefixes.wicReason, "no_types_for_object", ".")
    #     return False
    
    if len(objectCurTypes) > 0:
        # range types check
        for cls_ in objectCurTypes:
            if isSubClassOfAny(cls_, objectConstraintClasses, cleanTaxonomyUp):
                return True
        return False
    
    # writerMetaMessages.write("<<",s, p, o, ">>", Prefixes.wicReason, "range_constraint_not_satisfied", ".")
    return True


def isValidDomainRange(triple, subjectCurTypes, objectCurTypes, subjectConstraintClasses, objectConstraintClasses, cleanTaxonomyUp, writerMetaMessages):
    s, p, o = triple
    # statitics for both domain and range no_type count
    if not isValidDomain_noType(triple, subjectCurTypes, subjectConstraintClasses, cleanTaxonomyUp):
        if not isValidRange_noType(triple, objectCurTypes, objectConstraintClasses, cleanTaxonomyUp):
            writerMetaMessages.write("<<",s, p, o, ">>", Prefixes.wicReason, "no_types_for_domain_and_range", ".")
    # statitics for both domain and range violate count
    if not isValidDomain_violate(triple, subjectCurTypes, subjectConstraintClasses, cleanTaxonomyUp):
        if not isValidRange_violate(triple, objectCurTypes, objectConstraintClasses, cleanTaxonomyUp):
            writerMetaMessages.write("<<",s, p, o, ">>", Prefixes.wicReason, "both_domain_and_range_constraint_not_satisfied", ".")
    
    return True


def keepFactsForPredicates(triple, predicates_kept, writer):
    s, p, o = triple
    if p in predicates_kept:
        writer.write(s, p, o, ".")
        return True
    else:
        return False




##########################################################################
#             Main
##########################################################################

if __name__ == "__main__":
    with TsvUtils.Timer("Constrainting facts..."):
        # loading constraints
        constraints = load_constraints(SUBJ_CSV, VALUE_CSV)

        # loading taxonomy
        wicleanTaxonomyUp = dict()
        with open(TAXONOMY_PATH, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split(",")
                    if len(parts) == 2:
                        child, parent = parts[0], parts[1]
                        if ("wd:" + child) not in wicleanTaxonomyUp:
                            wicleanTaxonomyUp["wd:" + child] = set()
                        wicleanTaxonomyUp["wd:" + child].add("wd:" + parent)
        wicleanTaxonomyUp["wd:Q35120"] = set()  # add root node


        # loading instance types
        instDirectTypes = defaultdict(set)
        for split in TsvUtils.tsvTuples(INST_TYPE_PATH, "  Loading instance types..."):
            inst, rel, cls, _ = split
            if rel != Prefixes.wikidataType:
                continue
            if inst not in instDirectTypes:
                instDirectTypes[inst] = set()
            instDirectTypes[inst].add(cls)
        
        # initialize the writer
        writerMetaMessages = TsvUtils.TsvFileWriter(os.path.join(OUTPUT_FOLDER, META_MESSAGES_FILE))
        writerFacts = TsvUtils.TsvFileWriter(os.path.join(OUTPUT_FOLDER, FACTS_FILE))
        writerFacts.__enter__()
        writerMetaMessages.__enter__()
        n_facts_with_constraints = 0
        # type-checking
        for s,p,o,_ in TsvUtils.tsvTuples(FACTS_PATH, message="  Type-checking facts..."):
            if p in constraints:
                n_facts_with_constraints += 1
                relConstraint = constraints[p]
                triple = tuple((s,p,o))
                # Yiwen 05-23 no need after addition of occupation in taxonomy
                # if keepFactsForPredicates(triple, ["wdt:P106"], writerFacts): # keep all occupation facts
                    # continue
                # Do statitics count: no_valid_domian and no_valid_range
                isValidDomainRange(triple, instDirectTypes[s], instDirectTypes[o], relConstraint.subjectTypes, relConstraint.objectTypes, wicleanTaxonomyUp, writerMetaMessages)
                
                # domain check
                if not isValidDomain(triple, instDirectTypes[s], relConstraint.subjectTypes, wicleanTaxonomyUp, writerMetaMessages):
                    continue
                # range check
                if not NtUtils.isLiteral(o):
                    if not isValidRange(triple, instDirectTypes[o], relConstraint.objectTypes, wicleanTaxonomyUp, writerMetaMessages):
                        continue
                # write to file
                writerFacts.write(s, p, o, ".")
            else: # if p has no constraint, directly write to file
                writerFacts.write(s, p, o, ".")

        writerMetaMessages.__exit__()
        writerFacts.__exit__()
        print(" done")
        print(f"Number of facts with constraints: {n_facts_with_constraints}")


    # print(f"Loaded {len(constraints)} relation constraints.")
    # # Quick sanity check: print a few entries
    # for pid, rc in list(constraints.items())[5000:5005]:
    #     print(rc)
    #     print(f"  label       : {rc.rellabel}")
    #     print(f"  subjectTypes: {rc.subjectTypes}")
    #     print(f"  objectTypes : {rc.objectTypes}")
