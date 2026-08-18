'''
Extract initial taxonomy from Wikidata

CC-BY 2021 Fabian M. Suchanek

Note: 1) we extract the taxonomy from root class 'entity'(Q35120) by DFS traversal
      2) we also discard cycles and classes without labels during the extraction

'''

from typing import Any
import NtUtils
import TsvUtils
import Prefixes
import os
import graph_utils as utils
import csv
from collections import defaultdict
import networkx as nx
import config


###########################################################################
#           Loading the Wikidata taxonomy
###########################################################################


class wikidataVisitor(object):
    """ Will be called in parallel on each Wikidata entity graph, fills context[wikiTaxonomyDown]. """
    def __init__(self, id):
        self.wikidataTaxonomyDown={} # Direct subclasses
        self.wikiTaxonomyLabels={}
        self.wikiTaxonomyDescription={}

    def visit(self,graph): 
        predicates=graph.predicates()
        # Only care about the Taxonomy
        if Prefixes.wikidataSubClassOf not in predicates:
            return
        # Removing classes without labels
        if not Prefixes.rdfsLabel in predicates:
            return
        
        for s,p,o in graph:
            if p==Prefixes.rdfsLabel:
                self.wikiTaxonomyLabels[s]=o[:-3] # strip "@en"
            if p==Prefixes.schemaDescription: # Optional
                self.wikiTaxonomyDescription[s]=o[:-3] # strip "@en"
            if p==Prefixes.wikidataSubClassOf:
                if o not in self.wikidataTaxonomyDown:
                    self.wikidataTaxonomyDown[o]=set()
                self.wikidataTaxonomyDown[o].add(s)
    
    def result(self):
        return(self.wikidataTaxonomyDown, self.wikiTaxonomyLabels, self.wikiTaxonomyDescription)


###########################################################################
#           Cleaning the wikidata taxonomy
###########################################################################

class wikidataCleaner(object):
    """ Will be used for cleaning the built taxonomy """
    def __init__(self, cleanWikiTaxonomyDown: dict, cleanWikiTaxonomyUp: dict, wikiTaxonomyDown: dict,
                        valid_classes: set, metaclasses: set):
        # Used for outputs
        self.cleanWikiTaxonomyDown=cleanWikiTaxonomyDown
        self.cleanWikiTaxonomyUp=cleanWikiTaxonomyUp
        # Used for inputs
        self.wikiTaxonomyDown=wikiTaxonomyDown
        self.loopCounter = 0
        self.looplength = []
        self.classes = valid_classes
        self.metaclasses = metaclasses
        
        self.digraph = None
    
    def subClassInclude(self, superClass, potentialSubClass, path=[]):
        """TRUE if the subclasses of superClass include subClass"""
        if superClass==potentialSubClass:
            return True, path
        for subClass in self.cleanWikiTaxonomyDown.get(superClass,[]):
            newpath = path + [superClass] 
            loopDeleted, loopPath = self.subClassInclude(subClass, potentialSubClass, newpath)
            if loopDeleted:
                return True, loopPath
        return False, path
    

    def addSubClass(self, superClass, subClass):
        """Adds the Wikidata classes to the wiki clean taxonomy, excluding loops"""
        loopDeleted, loopPath = self.subClassInclude(subClass, superClass)
        if loopDeleted:
            self.loopCounter+=1
            loopLength = len(set(loopPath + [subClass, superClass]))
            self.looplength.append((set(loopPath + [subClass, superClass]), loopLength))
            return
        
        if subClass not in self.classes:
            # not a valid class
            return
        
        if subClass in self.metaclasses:
            # is a metaclass but not a first-order class
            # we discard metaclasses here for simplicity
            return
        
        self.cleanWikiTaxonomyUp[subClass].add(superClass)
        self.cleanWikiTaxonomyDown[superClass].add(subClass)
        # Avoid adding the subclasses again in case of double inheritance -> save time
        if subClass in self.cleanWikiTaxonomyDown:
            return
        for subClass2 in self.wikiTaxonomyDown.get(subClass,[]):    
            self.addSubClass(subClass, subClass2) 
    
    def results(self):
        # return self.cleanWikiTaxonomyDown, self.cleanWikiTaxonomyUp
        return self.digraph

    def byPassSomeClasses(self, clsSet: set):
        """By-passes the classes in the set"""
        if self.digraph is None:
            self.digraph = nx.DiGraph(self.cleanWikiTaxonomyDown)
        
        for cls_ in clsSet:
            if not self.digraph.has_node(cls_):
                continue
            # leaf nodes
            if self.digraph.out_degree(cls_) == 0:
                self.digraph.remove_node(cls_)
                continue
            # inner nodes
            children = list(self.digraph.successors(cls_))
            parents = list(self.digraph.predecessors(cls_))
            for pc in parents:
                for cc in children:
                    self.digraph.add_edge(pc, cc) # relink parent-child
            self.digraph.remove_node(cls_)

    
    def removeRareTopClasses(self, root: str):
        """Removes the top-1 classes with no subclasses"""
        topcls = list(self.digraph.successors(root)) # entity's top-level classes
        cache = set()
        for topcls_ in topcls:
            if self.digraph.out_degree(topcls_) == 0:
                self.digraph.remove_node(topcls_)
                cache.add(topcls_)
        return cache
    

    def removeScholarlyArticle(self, ScholarlyArticle: str):
        """Removes the ScholarlyArticle class"""
        clsDiscard = utils.getDescendants(ScholarlyArticle, self.cleanWikiTaxonomyDown)
        if self.digraph is None:
            self.digraph = nx.DiGraph(self.cleanWikiTaxonomyDown)
        for cls_ in clsDiscard:
            if not self.digraph.has_node(cls_):
                continue
            self.digraph.remove_node(cls_)
        return clsDiscard

    
    def removeNonInformativeClasses(self, cls_inst_count: dict):
        """Removes the non-informative classes"""
        n = self.digraph.number_of_nodes()
        nonInfoCls = utils.find_non_informative_cls(self.digraph, cls_inst_count)
        while len(nonInfoCls) > 0:
            # remove non-informative classes recursively
            for cls_ in nonInfoCls:
                parent = list(self.digraph.predecessors(cls_))[0]
                child = list(self.digraph.successors(cls_))[0]
                self.digraph.remove_node(cls_)
                if not nx.has_path(self.digraph, parent, child):
                    self.digraph.add_edge(parent, child)
            nonInfoCls = utils.find_non_informative_cls(self.digraph, cls_inst_count)
        clsRemoved = n - self.digraph.number_of_nodes()
        return clsRemoved
        
    
    def removeBFOClasses(self, bfo_classes: set, kept_classes: set, root: str):
        """Removes the BFO classes"""
        for cls_ in bfo_classes:
            if cls_ == 'wd:Q223557': # physical object
                continue
            if self.digraph.has_node(cls_):
                self.digraph.remove_node(cls_)
        # relink kept classes (manually)
        for cls_ in kept_classes: # role; occurance
            self.digraph.add_edge(root, cls_) 
        # filter the unreachable nodes
        non_reachable_nodes = set()
        all_nodes = set(nx.descendants(self.digraph, root))
        for node_ in self.digraph.nodes:
            if node_ == root:
                continue
            if node_ not in all_nodes:
                non_reachable_nodes.add(node_)
        for node_ in non_reachable_nodes:
            self.digraph.remove_node(node_)
        return len(non_reachable_nodes)+len(bfo_classes)-len(kept_classes)-1

    
    def removeInstanceLessClasses(self, cls_inst_count: dict):
        """Remove classes without cumulative instances"""
        txDown = nx.to_dict_of_lists(self.digraph)
        cls_no_inst = set()
        for cls_ in self.digraph.nodes:
            if cls_ in cls_inst_count:
                continue
            if utils.cumulative_stats_for_class(cls_, cls_inst_count, txDown) < 1:
                cls_no_inst.add(cls_)
        for cls_ in cls_no_inst:
            if not self.digraph.has_node(cls_):
                continue
            self.digraph.remove_node(cls_)
        return len(cls_no_inst)



if __name__ == '__main__':

    WIKIDATA_FILE = config.WIKIDATA_DUMP_FILE
    CLASSES_FILE = config.CLASSES_FILE
    METACLASSES_FILE = config.METACLASSES_FILE
    CLS_INST_COUNT_FILE = config.CLS_INST_COUNT_FILE
    BFO_CLASSES_FILE = config.BFO_CLASSES_FILE
    ScholarlyArticle = "wd:Q13442814"
    BFO_KEEP_CLASSES = {'wd:Q214339', 'wd:Q1190554'} # TBD: could be changed later on
    TAXONOMY_FILE = config.TAXONOMY_FILE
    TAXONOMY_LABELS_FILE = config.TAXONOMY_LABELS_FILE
    TAXONOMY_DESCRIPTIONS_FILE = config.TAXONOMY_DESCRIPTIONS_FILE

    # check if the wikidata dump exists
    if not os.path.exists(WIKIDATA_FILE):
        raise FileNotFoundError("Please first download the latest Wikidata dump \
                                from https://dumps.wikimedia.org/wikidatawiki/entities/ and place it in the folder 'data/wikidata/' and also decompress it.")
    
    # loading valid classes from classes.csv
    CLS_SET = set()
    with open(CLASSES_FILE, "r", encoding="utf-8") as f:
        reader = csv.reader(f) # "class"
        for row in reader:
            if row[0].startswith("http://www.wikidata.org/entity/"):
                CLS_SET.add('wd:' + row[0].split("/")[-1]) # it is the wd:QID
    print(f"  Info: Total number of valid classes: {len(CLS_SET)}")

    # loading metaclasses from metaclasses.csv
    METACLASSES_SET = set()
    with open(METACLASSES_FILE, "r", encoding="utf-8") as f:
        reader = csv.reader(f) # "instance", "metaclass"
        for row in reader:
            if row[0].startswith("http://www.wikidata.org/entity/"): 
                METACLASSES_SET.add('wd:' + row[0].split("/")[-1]) # it is the wd:QID
            if row[1].startswith("http://www.wikidata.org/entity/"): 
                METACLASSES_SET.add('wd:' + row[1].split("/")[-1]) # it is the wd:QID
    print(f"  Info: Total number of metaclasses: {len(METACLASSES_SET)}")

    # loading class instance count from class_instance_count.csv
    CLS_INST_COUNT = dict()
    with open(CLS_INST_COUNT_FILE, "r", encoding="utf-8") as f:
        reader = csv.reader(f) # "class", "instance_count"
        for row in reader:
            if row[0].startswith("http://www.wikidata.org/entity/"):
                cls = 'wd:' + row[0].split("/")[-1]
                if cls in CLS_SET and not cls in METACLASSES_SET:
                    CLS_INST_COUNT[cls] = int(row[1])

    # loading BFO classes from bfo_classes.csv
    BFO_CLASSES = set()
    with open(BFO_CLASSES_FILE, "r", encoding="utf-8") as f:
        reader = csv.reader(f) # "instance", "name"
        for row in reader:
            if row[0].startswith("http://www.wikidata.org/entity/"):
                BFO_CLASSES.add('wd:' + row[0].split("/")[-1]) # it is the wd:QID
    print(f"  Info: Total number of BFO classes: {len(BFO_CLASSES)}")


    ###########################################################################
    #           Extracting the taxonomy
    ###########################################################################
    # extract the taxonomy from the wikidata dump
    with TsvUtils.Timer("Creating Wiki taxonomy"):
        # Load Wikidata 
        results = NtUtils.visitWikidata(WIKIDATA_FILE, wikidataVisitor) # <results> is a list taxonomies, as we use multi-processing
        # We now merge them together in the global variable <wikidataTaxonomyDown> -> a dirty one
        wikidataTaxonomyDown, wikidataTaxonomyLabels, wikidataTaxonomyDescription = {}, {}, {}
        for result in results: # (taxonomy, labels, descriptions)
            for key in result[0]:
                if key not in wikidataTaxonomyDown:
                    wikidataTaxonomyDown[key]=set()
                wikidataTaxonomyDown[key].update(result[0][key])
            for s in result[1]:
                wikidataTaxonomyLabels[s]=result[1][s]
            for s in result[2]:
                wikidataTaxonomyDescription[s]=result[2][s]
        # print("  Info: Total number of Wikidata classes and taxonomic links", len(wikidataTaxonomyDown), " and ", sum(len(wikidataTaxonomyDown[s]) for s in wikidataTaxonomyDown))

        # Pre-processing the taxonomy from root node 'entity(wd:Q35120)'
        root = 'wd:Q35120' # entity
        cleanWikiTaxonomyDown=defaultdict(set)
        cleanWikiTaxonomyUp=defaultdict(set)
        topClasses=wikidataTaxonomyDown.get(root, []) # set of top-classes
        cleanWikiTaxonomyDown[root]=topClasses.copy()
        for c in topClasses:
            cleanWikiTaxonomyUp[c].add(root)
        # Also adding the root class to the TaxonomyUp
        cleanWikiTaxonomyUp[root]=set()
        wikidataTaxonomyLabels[root]='"entity"'
        wikidataTaxonomyDescription[root]='"anything that can be considered, discussed, or observed"'

        # Pre-cleaning the taxonomy
        cleaner=wikidataCleaner(cleanWikiTaxonomyDown, cleanWikiTaxonomyUp, wikidataTaxonomyDown,
                                CLS_SET, METACLASSES_SET)
        for topClass in topClasses: # dfs traversal
            for subclass in wikidataTaxonomyDown.get(topClass, []):
                cleaner.addSubClass(topClass, subclass)
        print("  Info: Loops removed:", cleaner.loopCounter)
        remScholarlyCls = cleaner.removeScholarlyArticle(ScholarlyArticle)
        print("  Info: ScholarlyArticle removed:", len(remScholarlyCls))
        # Skip-connection (bypass) for classes without descriptions
        clsNoDesc = set(cleaner.cleanWikiTaxonomyUp.keys()) - set(wikidataTaxonomyDescription.keys())
        cleaner.byPassSomeClasses(clsNoDesc)
        cleaner.removeRareTopClasses(root)
        remNonInfoCls = cleaner.removeNonInformativeClasses(CLS_INST_COUNT)
        print("  Info: Non-informative classes removed:", remNonInfoCls)
        
        remBFOCls = cleaner.removeBFOClasses(BFO_CLASSES, BFO_KEEP_CLASSES, root)
        print("  Info: Classes removed due to BFO classes:", remBFOCls)
        remInstanceLessCls = cleaner.removeInstanceLessClasses(CLS_INST_COUNT)
        print("  Info: Classes without cumulative instances removed:", remInstanceLessCls)


        # Check the statistics
        stats = {
            "number_of_nodes": cleaner.digraph.number_of_nodes(),
            "number_of_edges": cleaner.digraph.number_of_edges(),
            "max_depth": max(nx.shortest_path_length(cleaner.digraph, source=root).values()),
            "weakly_connected": nx.is_weakly_connected(cleaner.digraph),
            "directed_acyclic": nx.is_directed_acyclic_graph(cleaner.digraph),
            "number_of_roots": len([node for node in cleaner.digraph.nodes() if not list(cleaner.digraph.predecessors(node))]),
        }
        print('  ===================================================')
        for key, value in stats.items():
            print(f"  {key}: {value}")
        print('  ===================================================')
        
        # Write the initial taxonomy
        print("  Storing initial taxonomy...", end="", flush=True)
        with open(TAXONOMY_FILE, "w") as taxonomyWriter:
            for edge in cleaner.digraph.edges():
                parent, child = edge
                taxonomyWriter.write(f"{child[3:]},{parent[3:]}\n") # strip the wd: prefix
        
        with open(TAXONOMY_LABELS_FILE, "w") as taxonomyLabelWriter:
            for cls in cleaner.digraph.nodes():
                taxonomyLabelWriter.write(f"{cls[3:]}\t{wikidataTaxonomyLabels[cls]}\n") # strip the wd: prefix
        
        with open(TAXONOMY_DESCRIPTIONS_FILE, "w") as taxonomyDescriptionWriter:
            for cls in cleaner.digraph.nodes():
                if cls in wikidataTaxonomyDescription:
                    taxonomyDescriptionWriter.write(f"{cls[3:]}\t{wikidataTaxonomyDescription[cls]}\n") # strip the wd: prefix
                
        print("done")