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
from typing import Optional
import config
import csv

# Remove scholarly articles
PropertyToRemove = {
    # Special properties
    # "wdt:P106", # occupation -> serves as instance_of link
    # "wdt:P279", # subclass_of  -> as here we consider only instances, not class taxonomies
    # "wd:P31", # instance of -> already considered in ParseInstanceTypes.py
    Prefixes.skosPrefLabel,
    Prefixes.schemaName,
}


# Paths (defined in config.py)
DATA_PATH          = config.DATA_PATH
WIKIDATA_FILE      = config.WIKIDATA_FILE
FOLDER             = config.FACTS_FOLDER
META_MESSAGES_FILE = config.FACTS_META_MESSAGES_FILE
FACTS_FILE         = config.FACTS_FILE


# identifiers, otherwise too much facts
# Yiwen: later on .... in step 3: type check
identifierRelations = set()
with open(config.IDENTIFIERS_FILE, "r") as f:
    reader = csv.reader(f)
    for row in reader:
        if row[0].startswith("http://www.wikidata.org/entity/"):
            identifierRelations.add('wdt:' + row[0].split("/")[-1]) # it is the wdt:PID

# load non-labeled entities
nonLabeledEntities = set()
with open(config.NO_LABEL_INSTANCES_FILE, "r") as f:
    for line in f:
        if line.strip().startswith("wd:"):
            nonLabeledEntities.add(line.strip())

##########################################################################
#             Facts Grammer checks
##########################################################################

# Yiwen: All deleted, no need
# consider valid facts <s, p, o> where s, o has must have labels 
# relation must start with "wdt:" + some literal properties

##########################################################################
#             Facts Grammer checks
##########################################################################

def keepFactsForPredicates(entityFacts, predicates, writer):
    for p in predicates:
        for s, p, o in entityFacts.triplesWithPredicate(p):
            writer.write(s, p, o, ".")


##########################################################################
#             Remove additional facts
##########################################################################

def removeAdditionalFacts(entityFacts, predicates, writerMetaMessages):
    # remove facts with p in predicates
    # also remove facts with p.startswith("wdtn")
    # also remove rare cases s==o, e.g. wd:Q96935054
    for p in list(entityFacts.predicates()):
        if p == Prefixes.wikidataType: # P31 already considered in ParseInstanceTypes.py, so remove all
            for t in entityFacts.triplesWithPredicate(p):
                entityFacts.remove(t)
            continue
        if p == Prefixes.wikidataSubClassOf: # P279 already considered in TaxonomyCleaning, so remove all
            for t in entityFacts.triplesWithPredicate(p):
                entityFacts.remove(t)
            continue
        if p in predicates:
            for t in entityFacts.triplesWithPredicate(p):
                entityFacts.remove(t)
                writerMetaMessages.write("<<", t[0], p, t[2], ">>", Prefixes.wicReason, "redundant_facts", ".")
        elif p.startswith("wdtn"):
            for t in entityFacts.triplesWithPredicate(p):
                entityFacts.remove(t)
                writerMetaMessages.write("<<", t[0], p, t[2], ">>", Prefixes.wicReason, "meta_facts", ".")
        else:
            for t in entityFacts.triplesWithPredicate(p):
                if t[0] == t[2]:
                    entityFacts.remove(t)
                    writerMetaMessages.write("<<", t[0], p, t[2], ">>", Prefixes.wicReason, "rare_facts_s_o_equal", ".")


def excludeAllScholaryArticlesFacts(entityFacts, scholarlyArticleClasses):
    isScholaryArticle = False
    for t in entityFacts.triplesWithPredicate(Prefixes.wikidataType):
        if t[2] in scholarlyArticleClasses:
            isScholaryArticle = True
            break
    if isScholaryArticle:
        # count all facts for this entity
        count = entityFacts.numberOfTriples()
        return False, count
    return True, 0


def countIdentifierRelationFacts(entityFacts):
    # count = sum(1 for t in entityFacts.triplesWithPredicate(*identifierRelations))
    mainEntity = entityFacts.mainSubject()
    count = 0
    for predicate in entityFacts.predicates():
        if predicate in identifierRelations:
            count += len(entityFacts.objects(subject=mainEntity, predicate=predicate))
    return count


def removeIdentifierRelationFacts(entityFacts, identifierRelations):
    for predicate in entityFacts.predicates():
        if predicate in identifierRelations:
            for t in entityFacts.triplesWithPredicate(predicate):
                entityFacts.remove(t)


def removeNonLabeledFacts(entityFacts, writerMetaMessages):
    ToRemove = []
    for t in entityFacts:
        s, p, o = t
        if s in nonLabeledEntities or o in nonLabeledEntities:
            ToRemove.append(t)
    for t_ in ToRemove:
        entityFacts.remove(t_)
        s, p, o = t_
        writerMetaMessages.write("<<", s, p, o, ">>", Prefixes.wicReason, "non_labeled_subject_or_object", ".")


##########################################################################
#             Filtering literals
##########################################################################
def normalizeString(s) -> Optional[str]:
    """ Makes sure that a string does not contain invalid characters or languages"""
    if not s or not s.startswith('"'):
        return s
    return s.replace("\uFFFD", "_").replace('"@zh-classical', '"@zh')


def onlyKeepEnglishLiterals(entityFacts):
    ToRemove = []
    for t in entityFacts:
        # deal with the literals
        if NtUtils.isLiteral(t[2]):
            _, _, lang, _ = NtUtils.splitLiteral(t[2])
            if lang and lang != "en":
                ToRemove.append(t)
    for t in ToRemove:
        entityFacts.remove(t)

##########################################################################
#             Main method
##########################################################################

class treatWikidataEntity():
    """ Visitor that will handle every Wikidata entity """
    def __init__(self,i):
        """ We load everything once per process (!) in order to avoid problems with shared memory """
        print("    Initializing Wikidata reader",i+1, flush=True)
        self.number=i

        # load scholarly article classes and their descendants
        self.scholarlyArticleClasses = set()
        with open(config.SCHOLARLY_ARTICLE_CLASSES_FILE, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                if row[0].startswith("http://www.wikidata.org/entity/"):
                    self.scholarlyArticleClasses.add('wd:' + row[0].split("/")[-1]) # it is the wd:QID

        print("    Done initializing Wikidata reader",i+1, flush=True)
        self.writer=None
        

    def visit(self,entityFacts):
        """ Writes out the facts for a single Wikidata entity """
        
        # We have to open the file here and not in init() to avoid pickling problems
        if not self.writer:
            self.writer=TsvUtils.TsvFileWriter(FOLDER+"wiki_facts"+(str(self.number).rjust(4,'0'))+".tmp")
            self.writer.__enter__()
        
        # Anything that is rdf:type in Wikidata is meta-statements, 
        # and should go away
        for t in entityFacts.triplesWithPredicate(Prefixes.rdfType):
            entityFacts.remove(t)
        
        # remove scholarly article facts
        excludeScholaryArticle, factsCount = excludeAllScholaryArticlesFacts(entityFacts, self.scholarlyArticleClasses)
        if not excludeScholaryArticle:
            self.writer.write("**FactsCountForScholaryArticle", str(factsCount)) # To remove: Just to calculate the statistics
            return
        
        # if no labels, return
        if not entityFacts.triplesWithPredicate(Prefixes.rdfsLabel):
            for s,p,o in entityFacts:
                self.writer.write("<<", s, p, o, ">>", Prefixes.wicReason, "non_labeled_subject_or_object", ".")
            return
        
        # remove non-labeled facts
        removeNonLabeledFacts(entityFacts, self.writer)
        
        # remove additional facts
        removeAdditionalFacts(entityFacts, PropertyToRemove, self.writer)

        # only keep english literals
        onlyKeepEnglishLiterals(entityFacts)

        # count identifier relation facts
        n_identifierRelationFacts = countIdentifierRelationFacts(entityFacts)
        if n_identifierRelationFacts > 0:
            self.writer.write("**FactsCountForIdentifierRelation", str(n_identifierRelationFacts)) # To remove: Just to calculate the statistics
        
        # remove the facts with identifer relations
        removeIdentifierRelationFacts(entityFacts, identifierRelations)

        # Write out the facts
        for s,p,o in entityFacts:
            # Deal with special cases where '\n' exist in math expressions
            # e.g. <http://www.wikidata.org/entity/Q123024376> <http://www.wikidata.org/prop/direct/P7235> 
            # "<math xmlns=\"http://www.w3.org/1998/Math/MathML\" display=\"block\" alttext=\"{\\displaystyle \\Theta }\">\n "^^<http://www.w3.org/1998/Math/MathML> .
            o = o.replace("\n", "\\n") 
            if s==o:
                # Rare cases that are nonsense, e.g. wd:Q96935054
                continue
            self.writer.write(s,p,o,".")

    def result(self):
        self.writer.__exit__()
        return None



if __name__ == '__main__':

    # check if the wikidata dump exists
    if not os.path.exists(WIKIDATA_FILE):
        raise FileNotFoundError("Please first download the latest Wikidata dump \
                                from https://dumps.wikimedia.org/wikidatawiki/entities/ and place it in the folder 'data/wikidata/' and also decompress it.")

    with TsvUtils.Timer("Extracting Wikidata facts"):
        NtUtils.visitWikidata(WIKIDATA_FILE, treatWikidataEntity, numThreads=65)
        print("  Collecting results...")
        count=0
        allScholaryArticleFacts=0
        allIdentifierRelationFacts=0
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
                            elif line.startswith(b"**FactsCountForScholaryArticle"):
                                allScholaryArticleFacts+=int(line.split(b'\t')[1])
                            elif line.startswith(b"**FactsCountForIdentifierRelation"):
                                allIdentifierRelationFacts+=int(line.split(b'\t')[1]) # statistics, not deleted in reality
                            elif line.strip():
                                writer.write(line)
                                count+=1
        print("  done")
        print("===================================================")
        print("  Info: Number of facts counted:", count)
        print("  Info: Number of facts for ScholaryArticle (removed):", allScholaryArticleFacts)
        print("  Info: Number of facts for IdentifierRelation (removed):", allIdentifierRelationFacts)
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