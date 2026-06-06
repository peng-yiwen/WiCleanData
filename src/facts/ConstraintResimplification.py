from FactsTypeCheck import RelationConstraint
from FactsTypeCheck import load_constraints
import csv
import TsvUtils
import Prefixes
import NtUtils
from collections import defaultdict
import config


# Paths (defined in config.py)
SUBJ_CSV         = config.RESIMP_SUBJ_CSV
VALUE_CSV        = config.RESIMP_VALUE_CSV
TAXONOMY_PATH    = config.TAXONOMY_FILE
INST_TYPE_PATH   = config.RESIMP_INST_TYPE_PATH
FACTS_PATH       = config.RESIMP_FACTS_PATH
UNUSED_TYPES_CSV = config.RESIMP_UNUSED_TYPES_CSV
IDENTIFIER_PATH = config.IDENTIFIERS_FILE



##########################################################################
#             Useful functions
##########################################################################
def load_identifiers(path):
    identifierRelations = set()
    with open(path, "r") as f:
        for line in f:
            identifierRelations.add(line.strip())
    return identifierRelations


def getSuperClasses(cls, classes, cleanWikiTaxonomyUp):
    """Adds all superclasses of a class <cls> (including <cls>) to the set <classes>"""
    classes.add(cls)
    # Make a check before because it's a defaultdict,
    # which would create cls if it's not there
    if cls in cleanWikiTaxonomyUp:
        for sc in cleanWikiTaxonomyUp[cls]:
            getSuperClasses(sc, classes, cleanWikiTaxonomyUp)      


def getClasses(directClasses, cleanWikiTaxonomyUp):
    """Returns the set of all classes and their superclasses that the subject is an instance of"""
    classes=set()
    for directClass in directClasses:
        getSuperClasses(directClass, classes, cleanWikiTaxonomyUp)        
    return classes


##########################################################################
#             Main
##########################################################################

if __name__ == "__main__":
    with TsvUtils.Timer("Constrainting facts..."):
        # loading constraints
        constraints = load_constraints(SUBJ_CSV, VALUE_CSV)
        # identifierRelations = load_identifiers(IDENTIFIER_PATH)

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
    
    
        # type-checking
        existingRelations = set()
        for s,p,o,_ in TsvUtils.tsvTuples(FACTS_PATH, message="  Counting constraint types..."):
            if p == "wdt:P106":
                # existingRelations.add(p) # suppose all is used
                continue
            if p in constraints:
                existingRelations.add(p)
                relConstraint = constraints[p]
                triple = tuple((s,p,o))

                if len(relConstraint.subjectTypes) > 0:
                    fullSubjectTypes = getClasses(instDirectTypes[s], wicleanTaxonomyUp)
                    overlap = fullSubjectTypes & relConstraint.subjectTypes
                    for cls_ in overlap:
                        relConstraint.subjectTypesCount[cls_] += 1
                
                if len(relConstraint.objectTypes) > 0:
                    fullObjectTypes = getClasses(instDirectTypes[o], wicleanTaxonomyUp)
                    overlap = fullObjectTypes & relConstraint.objectTypes
                    for cls_ in overlap:
                        relConstraint.objectTypesCount[cls_] += 1

        # output relation with its unused subject or object types (count == 0)
        with open(UNUSED_TYPES_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["relation", "label", "types_unused"])
            for pid in sorted(constraints.keys()):
                if pid not in existingRelations: # avoid some identifier relations
                    continue
                rc = constraints[pid]
                if not rc.subjectTypes and not rc.objectTypes:
                    continue
                unused_subj = sorted(
                    cls_ for cls_ in rc.subjectTypes
                    if rc.subjectTypesCount[cls_] == 0
                )
                unused_obj = sorted(
                    cls_ for cls_ in rc.objectTypes
                    if rc.objectTypesCount[cls_] == 0
                )
                if not unused_subj and not unused_obj:
                    continue
                types_unused = (
                    f"subject:[{','.join(unused_subj)}], "
                    f"object:[{','.join(unused_obj)}]"
                )
                writer.writerow([pid, rc.rellabel, types_unused])

        print(" done")




    # print(f"Loaded {len(constraints)} relation constraints.")
    # # Quick sanity check: print a few entries
    # for pid, rc in list(constraints.items())[5000:5005]:
    #     print(rc)
    #     print(f"  label       : {rc.rellabel}")
    #     print(f"  subjectTypes: {rc.subjectTypes}")
    #     print(f"  objectTypes : {rc.objectTypes}")
