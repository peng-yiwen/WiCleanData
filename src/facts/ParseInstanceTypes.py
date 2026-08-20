'''
Modified version of the facts generator from Wikidata facts.

Original author: Fabian M. Suchanek
Original source: https://github.com/yago-naga/yago-4.5/blob/main/03-make-facts.py
License: CC-BY 4.0 International License
'''

import Prefixes
import glob
import TsvUtils
import NtUtils
import os
import config
import csv

# Remove scholarly articles
PropertyToRemove = {
    # Special properties
    "wdt:P279", # subclass_of  -> as here we consider only instances, not class taxonomies
    Prefixes.skosPrefLabel,
    Prefixes.schemaName,
}

# Paths (defined in config.py)
DATA_PATH          = config.DATA_PATH
WIKIDATA_FILE      = config.WIKIDATA_FILE
FOLDER             = config.INST_TYPES_FOLDER
META_MESSAGES_FILE = config.INST_META_MESSAGES_FILE
FACTS_FILE         = config.INST_FACTS_FILE



##########################################################################
#             Checks
##########################################################################

def onlykeepFacts(entityFacts, predicates):
    """ Only keep the facts with the given predicates """
    for p in entityFacts.predicates():
        if p not in predicates:
            for t in entityFacts.triplesWithPredicate(p):
                entityFacts.remove(t)


def isValidInstance(entityFacts, classes, writerMetaMessages):
    """ Check if the subject is a valid instance (if has labels and descriptions, if instance is not a class) """
    mainEntity = entityFacts.mainSubject()

    # Must have a label
    if not entityFacts.triplesWithPredicate(Prefixes.rdfsLabel):
        for t in entityFacts.triplesWithPredicate(Prefixes.wikidataType):
            writerMetaMessages.write("<<", mainEntity, Prefixes.wikidataType, t[2], ">>", Prefixes.wicReason, "instance_no_label", ".")
        return False

    # Must have a description
    # if not entityFacts.triplesWithPredicate(Prefixes.schemaDescription):
    #     for t in entityFacts.triplesWithPredicate(Prefixes.wikidataType):
    #         writerMetaMessages.write("<<", mainEntity, Prefixes.wikidataType, t[2], ">>", Prefixes.wicReason, "instance_no_description", ".")
    #     return False

    # # Must not be a class
    # if mainEntity in classes:
    #     for t in entityFacts.triplesWithPredicate(Prefixes.wikidataType):
    #         writerMetaMessages.write("<<", mainEntity, Prefixes.wikidataType, t[2], ">>", Prefixes.wicReason, "instance_is_a_class", ".")
    #     return False

    return True



def isValidObjectTypes(entityFacts, classes, writerMetaMessages):
    """ Check if the object is a valid type (if is a class) """
    mainEntity = entityFacts.mainSubject()

    # Remove wikidataType triples whose object is not a valid class
    for t in entityFacts.triplesWithPredicate(Prefixes.wikidataType):
        if t[2] not in classes:
            writerMetaMessages.write("<<", mainEntity, Prefixes.wikidataType, t[2], ">>", Prefixes.wicReason, "type_"+t[2]+"_not_a_valid_class", ".")
            entityFacts.remove(t)

    # At least one valid type must remain
    return bool(entityFacts.triplesWithPredicate(Prefixes.wikidataType))



##########################################################################
#             Retype Instance Types
##########################################################################

def retypeInstancesToClean(entityFacts, wicleanmap, wicleanTaxonomyUp, wikipediaClasses, writerMetaMessages):
    mainEntity = entityFacts.mainSubject()
    for t in entityFacts.triplesWithPredicate(Prefixes.wikidataType):
        originalType = t[2]
        mappedType = originalType

        # first check if types in wicleanmap, if yes, replace with the mapped class
        # follow the chain until the terminal mapped class (guard against cycles with visited set)
        visited = set()
        while mappedType in wicleanmap and mappedType not in visited:
            visited.add(mappedType)
            mappedType = wicleanmap[mappedType]

        # replace the type with the mapped type
        if mappedType != originalType:
            entityFacts.remove(t)
            entityFacts.add((mainEntity, Prefixes.wikidataType, mappedType))
            assert t[0] == mainEntity # check if the subject is the same

        # check if type in wicleanTaxonomyUp, if not, post meta message and remove
        if mappedType not in wicleanTaxonomyUp:
            writerMetaMessages.write("<<", mainEntity, Prefixes.wikidataType, mappedType, ">>", Prefixes.wicReason, "type_"+mappedType+"_being_deleted_during_taxonomy_cleaning", ".")
            entityFacts.remove((t[0], Prefixes.wikidataType, mappedType))
    
    # check if no types left
    if len(entityFacts.triplesWithPredicate(Prefixes.wikidataType)) == 0:
        return 

    # remove transitive types
    fullTransitiveTypes = getClasses(entityFacts, wicleanTaxonomyUp)
    removeRedundantDirectClasses(entityFacts, fullTransitiveTypes, wicleanTaxonomyUp, writerMetaMessages)

    # retype to wikipediaClasses
    for t in entityFacts.triplesWithPredicate(Prefixes.wikidataType):
        cur_type = t[2]
        if cur_type in wikipediaClasses:
            continue
        # find the closest ancestor in wikipediaClasses when not in wikipediaClasses
        closest_ancestor = getFirstSuperClasses(cur_type, wicleanTaxonomyUp, wikipediaClasses)
        if len(closest_ancestor) == 0:
            writerMetaMessages.write("<<", t[0], Prefixes.wikidataType, t[2], ">>", Prefixes.wicReason, "type_"+cur_type+"_lost_after_wikipedia_filtering", ".")
            entityFacts.remove(t)
            continue
        closest_ancestors = sorted(closest_ancestor, key=lambda x: x[1])
        # only keep the closest ancestor one (i.e., min distance),
        # as we already go up to find classes, it definitely contains the general info of the original type
        min_depth = closest_ancestors[0][1]
        find_ancestor = False
        for ancestor_, depth in closest_ancestors:
            # skip root class
            if ancestor_ == 'wd:Q35120':  # root
                continue
            if depth > min_depth:
                break
            # add type facts
            entityFacts.add((t[0], Prefixes.wikidataType, ancestor_))
            find_ancestor = True
        entityFacts.remove(t)
        if not find_ancestor:
            writerMetaMessages.write("<<", t[0], Prefixes.wikidataType, t[2], ">>", Prefixes.wicReason, "type"+cur_type+"_lost_after_wikipedia_filtering", ".")
            continue

    # remove transitive types again
    fullTransitiveTypes = getClasses(entityFacts, wicleanTaxonomyUp)
    removeRedundantDirectClasses(entityFacts, fullTransitiveTypes, wicleanTaxonomyUp, writerMetaMessages)



##########################################################################
#             Useful functions
##########################################################################

def getValidAncestors(cls, ancestors, oriTaxonomyUp, clean_classes, depth):
    """
    Depth: means how far the ancestor is from the current class
    """
    for sp in oriTaxonomyUp[cls]:
        # path should not include irrelevant edges
        if sp in clean_classes:
            ancestors.add(tuple([sp, depth]))
            continue
        getValidAncestors(sp, ancestors, oriTaxonomyUp, clean_classes, depth+1)


def getFirstSuperClasses(cls, oriTaxonomyUp, clean_classes):
    ancestors = set()
    getValidAncestors(cls, ancestors, oriTaxonomyUp, clean_classes, depth=0)
    return ancestors


def getSuperClasses(cls, classes, cleanWikiTaxonomyUp):
    """Adds all superclasses of a class <cls> (including <cls>) to the set <classes>"""
    classes.add(cls)
    # Make a check before because it's a defaultdict,
    # which would create cls if it's not there
    if cls in cleanWikiTaxonomyUp:
        for sc in cleanWikiTaxonomyUp[cls]:
            getSuperClasses(sc, classes, cleanWikiTaxonomyUp)      


def getClasses(entityFacts, cleanWikiTaxonomyUp):
    """Returns the set of all classes and their superclasses that the subject is an instance of"""
    classes=set()
    for directClass in entityFacts.objects(None, Prefixes.wikidataType):
        getSuperClasses(directClass, classes, cleanWikiTaxonomyUp)        
    return classes


def removeRedundantDirectClasses(entityFacts, fullTransitiveClasses, cleanWikiTaxonomyUp, writerMetaMessages):
    """ Removes all redundant classes among the entity facts. 
        Only keep the direct superclasses, removing the transitive ones, which are redundant. """
    for t in entityFacts.triplesWithPredicate(Prefixes.wikidataType):
        if any(t[2] in cleanWikiTaxonomyUp[c] for c in fullTransitiveClasses):
            entityFacts.remove(t)
            writerMetaMessages.write("<<", t[0], Prefixes.wikidataType, t[2], ">>", Prefixes.wicReason, "is_a_shortcut", ".")



##########################################################################
#             Main method
##########################################################################

class treatWikidataEntity():
    """ Visitor that will handle every Wikidata entity """
    def __init__(self,i):
        """ We load everything once per process (!) in order to avoid problems with shared memory """
        print("    Initializing Wikidata reader",i+1, flush=True)
        self.number=i
        
        print("    Wikidata reader", i+1, "loads resources", flush=True)
        # load valid classes after extraction
        self.isClasses = set()
        with open(config.VALID_CLASSES_FILE, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                if row[0].startswith("http://www.wikidata.org/entity/"):
                    self.isClasses.add('wd:' + row[0].split("/")[-1]) # it is the wd:QID
        
        # load wiclean taxonomy (before Wikipedia filtering)
        self.wicleanTaxonomyUp = dict()
        with open(config.TAXONOMY_BEFORE_WP_FILE, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split(",")
                    if len(parts) == 2:
                        child, parent = parts[0], parts[1]
                        if ('wd:' + child) not in self.wicleanTaxonomyUp:
                            self.wicleanTaxonomyUp['wd:' + child] = set()
                        self.wicleanTaxonomyUp['wd:' + child].add('wd:' + parent)

        # load all classes from wiclean taxonomy (after Wikipedia filtering)
        self.wikipediaClasses = set()
        with open(config.TAXONOMY_FILE, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    for part in line.split(","):
                        self.wikipediaClasses.add('wd:' + part)

        # load wiclean mapping (original -> current class)
        self.wicleanMap = {}
        with open(config.WICLEAN_MAPPING_FILE, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split(",")
                    if len(parts) == 2:
                        self.wicleanMap['wd:' + parts[0]] = 'wd:' + parts[1]
        
        root = 'wd:Q35120' # entity
        self.wicleanTaxonomyUp[root] = set()
        self.writer=None
    
        
    def visit(self, entityFacts):
        """ Writes out the InstanceTypes for a single Wikidata entity """

        # We have to open the file here and not in init() to avoid pickling problems
        if not self.writer:
            self.writer=TsvUtils.TsvFileWriter(FOLDER+"wiki_facts"+(str(self.number).rjust(4,'0'))+".tmp")
            self.writer.__enter__()
        
        # Only consider entities with wikidataType
        if not entityFacts.triplesWithPredicate(Prefixes.wikidataType):
            mainEntity = entityFacts.mainSubject()
            # further note the instances without labels
            if not entityFacts.triplesWithPredicate(Prefixes.rdfsLabel):
                self.writer.write("<<", mainEntity, Prefixes.wikidataType, "None_Type", ">>", Prefixes.wicReason, "instance_no_label", ".")
            return

        # Anything that is rdf:type in Wikidata is meta-statements, 
        # and should go away
        for t in entityFacts.triplesWithPredicate(Prefixes.rdfType):
            entityFacts.remove(t)
        
        # Remove all facts except for rdfsLabel, schemaDescription, wikidataType
        onlykeepFacts(entityFacts, [Prefixes.rdfsLabel, 
                                    Prefixes.schemaDescription, 
                                    Prefixes.wikidataType])

        # check if the subject is a valid instance (literals; not a class)
        if not isValidInstance(entityFacts, self.isClasses, self.writer):
            return

        # check if the wikidataTypes is valid
        if not isValidObjectTypes(entityFacts, self.isClasses, self.writer):
            return

        # retype types to wikipedaClasses based on the wicleanTaxonomyUp
        retypeInstancesToClean(entityFacts, self.wicleanMap, self.wicleanTaxonomyUp, self.wikipediaClasses, self.writer)
        if not entityFacts.triplesWithPredicate(Prefixes.wikidataType):
            return

        # write out the InstanceTypes
        for s,p,o in entityFacts:
            # Deal with special cases where '\n' exist in math expressions
            # e.g. <http://www.wikidata.org/entity/Q123024376> <http://www.wikidata.org/prop/direct/P7235> 
            # "<math xmlns=\"http://www.w3.org/1998/Math/MathML\" display=\"block\" alttext=\"{\\displaystyle \\Theta }\">\n "^^<http://www.w3.org/1998/Math/MathML> .
            o = o.replace("\n", "\\n") 
            if s==o:
                # Rare cases that are nonsense, e.g. wd:Q96935054
                continue
            if p == Prefixes.wikidataType:
                self.writer.write(s,p,o,".")

    def result(self):
        self.writer.__exit__()
        return None



if __name__ == '__main__':

    # check if the wikidata dump exists
    if not os.path.exists(WIKIDATA_FILE):
        raise FileNotFoundError("Please first download the latest Wikidata dump \
                                from https://dumps.wikimedia.org/wikidatawiki/entities/ and place it in the folder 'data/wikidata/' and also decompress it.")

    with TsvUtils.Timer("Extracting Wikidata Instance Types"):
        NtUtils.visitWikidata(WIKIDATA_FILE, treatWikidataEntity, numThreads=65) # contain wd: prefix
        print("  Collecting results...")
        count=0
        tempFiles=list(glob.glob(FOLDER+"wiki_facts*.tmp"))
        tempFiles.sort()
        with open(FOLDER+META_MESSAGES_FILE, "wb") as logwriter:
            with open(FOLDER+FACTS_FILE, "wb") as writer:
                for file in tempFiles:
                    print("    Reading",file)
                    with open(file, "rb") as reader:
                        for line in reader:
                            if line.startswith(b"<<"):
                                logwriter.write(line) # meta messages
                            elif line.strip():
                                writer.write(line)
                                count+=1
        print("  done")
        print("===================================================")
        print("  Info: Number of type facts:", count)
        print("===================================================")
        
        print("  Deleting temporary files...", end="", flush=True)
        for file in tempFiles:
            os.remove(file)
        print(" done")
    
    # # Calculate the statistics
    # print("Calculating Statistics...", end="", flush=True) 
    # total_cls_counts = utils.ent_mentions(os.path.join(FOLDER, "wiki_taxonomy.tsv"))
    # stats_prop = utils.prop_mentions(os.path.join(FOLDER, "wiki_facts.tsv"))
    # stats_ent = utils.ent_mentions(os.path.join(FOLDER, "wiki_facts.tsv"))
    # cls_inst_stats = utils.cls_mentions(os.path.join(FOLDER, "wiki_facts.tsv"))
    # stats_typed_inst = utils.inst_type_mentions(os.path.join(FOLDER, "wiki_facts.tsv"))

    # n_cls_total = len(total_cls_counts.keys())
    # n_cls_with_insts = len(cls_inst_stats.keys())
    # n_cls_without_insts = n_cls_total - n_cls_with_insts
    # n_typed_insts = len(stats_typed_inst.keys())
    # n_insts = len(stats_ent.keys())
    # n_props = len(stats_prop.keys())
    # n_facts = sum(stats_prop.values())

    # with open(FOLDER+"statistics_info_ParseWikiFacts.tsv", "w") as writer:
    #     writer.write("****Wikidata statistics****\n")
    #     writer.write("Number of classes:\t"+str(n_cls_total)+"\n")
    #     writer.write("Number of entities (sujects & objects):\t"+str(n_insts)+"\n")
    #     writer.write("Number of predicates:\t"+str(n_props)+"\n")
    #     writer.write("Number of facts:\t"+str(n_facts)+"\n")
    #     writer.write("Number of classes without direct instances:\t"+str(n_cls_without_insts)+"\n")
    #     writer.write("Number of classes having direct instances:\t"+str(n_cls_with_insts)+"\n")
    #     writer.write("Number of typed instances:\t"+str(n_typed_insts)+"\n")
