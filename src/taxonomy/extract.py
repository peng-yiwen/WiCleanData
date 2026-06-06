import networkx as nx
import TsvUtils
import Prefixes
import utils
import os
from collections import defaultdict
from tqdm import tqdm
import graph_utils
import pandas as pd


class wikidataCleaner(object):
    """ Will be used for cleaning the built taxonomy """
    def __init__(self, cleanWikiTaxonomyDown: dict, cleanWikiTaxonomyUp: dict, wikiTaxonomyDown: dict):
        # Used for outputs
        self.cleanWikiTaxonomyDown=cleanWikiTaxonomyDown
        self.cleanWikiTaxonomyUp=cleanWikiTaxonomyUp
        # Used for inputs
        self.wikiTaxonomyDown=wikiTaxonomyDown
        self.loopCounter = 0
        self.looplength = []
    
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
        
        # if subClass not in CLS_SET:
        #     # not a valid class
        #     return
        self.cleanWikiTaxonomyUp[subClass].add(superClass)
        self.cleanWikiTaxonomyDown[superClass].add(subClass)
        # Avoid adding the subclasses again in case of double inheritance -> save time
        if subClass in self.cleanWikiTaxonomyDown:
            return
        for subClass2 in self.wikiTaxonomyDown.get(subClass,[]):    
            self.addSubClass(subClass, subClass2) 


def load_literals(path):
    cls2label = {} # qid: label
    with open(path, 'r') as f_label:
        for line in f_label:
            terms = line.strip().split('\t')
            if len(terms) >= 2:
                cls2label[terms[0]] = terms[1][1:-1]
    return cls2label

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
        taxonomyDown (dict): Taxonomy from top to down.
    """
    descendants = getDescendants(cls, taxonomyDown) # including cls itself
    return sum(stats.get(descendant, 0) for descendant in descendants)


def find_non_informative_cls(graph, cls_stats):
    '''
    Find classes with only one superclass, one subclass, and without direct instances
    @param graph: networkx.DiGraph
    @param cls_stats: dict, class:direct_instance_count
    '''
    redundant_nodes = set()
    for node in graph.nodes():
        if graph.in_degree(node) == 1 and graph.out_degree(node) == 1:
            stats = cls_stats.get(node, 0) # ensure the node is in cls_stats
            if stats < 1:
                redundant_nodes.add(node)
    return redundant_nodes



def remove_rare_classes(graph, wikipedia_loc, root='Q35120', depth=3):
    ''' 
    Removes classes that (1) either without cumulative instances (include itself)
        or (2) without a Wikipedia page (multiple languages)

    Args:
        wikipedia_loc (str): Path to the Wikipedia mapped entities file.
        depth (int): Depth to keep for top-level classes.
    '''
    # # (1) filter classes without cumulative instances
    # cum_stats = graph_utils.cumulative_stats(self.cls_inst_stats, nx.to_dict_of_lists(self.wiki_dag.reverse()))
    # nodes = list(self.wiki_dag.nodes())
    # longtail_count = 0
    # for cls in tqdm(nodes, desc='Filter classes without cumulative instances'):
    #     if cum_stats[cls] < 1:
    #         self.wiki_dag.remove_node(cls)
    #         self.cls_inst_stats.pop(cls, None)
    #         longtail_count += 1
    # print(f"    Removed {longtail_count} classes without any cumulative instances.")
    
    # (2) filter classes without a Wikipedia page
    mapped_ents = graph_utils.load_wikipedia_mapped_ents(graph, wikipedia_loc) # has wd:
    if len(mapped_ents) == 0:
        raise ValueError("No Wikipedia page found for the graph")
    topcls = nx.single_source_shortest_path_length(graph, source=root, cutoff=depth).keys()
    mapped_ents.update(topcls)
    mapped_wiki_ents = mapped_ents & set(graph.nodes())
    if len(mapped_wiki_ents) == 0:
        raise ValueError("No Wikipedia page found for the graph...")
    # rebuild graph based on mapped wiki classes
    wikc = nx.DiGraph()
    for node in mapped_wiki_ents:
        wikc.add_node(node)
    for node in tqdm(mapped_wiki_ents, desc='Filter classes without a Wikipedia page'):
        ancestors = list(graph_utils.get_first_ancestors_for_rebuild(graph, node, mapped_wiki_ents))
        for ancestor in ancestors:
            wikc.add_edge(ancestor, node) # parent -> child
    # wikc = nx.transitive_reduction(wikc)

    return wikc

if __name__ == "__main__":

    # Class to discard
    ScholarlyArticle = "Q13442814"
    PATH = '../../data/data_2026/'
    # CLS_SET = utils.read_cls(os.path.join(PATH, "instORcls.tsv"))

    # # Differentiate the classes from instances
    # oriwikiDown, oriwikiUp = utils.load_taxonomy(os.path.join(PATH, "wiki_taxonomy.tsv"))
    # wikiTaxonomyDown, wikiTaxonomyUp = defaultdict(set), defaultdict(set)
    # root = 'wd:Q35120' # entity
    # topClasses = oriwikiDown.get(root, []) # set of top-classes
    # wikiTaxonomyDown[root] = topClasses.copy()
    # wikiTaxonomyUp[root] = set()
    # for c in topClasses:
    #     wikiTaxonomyUp[c].add(root)
    
    # for topClass in topClasses: # DFS traversal
    #     for subclass in oriwikiDown.get(topClass, []):
    #         addSubClass(topClass, subclass)

    # first load the taxonomy
    wikiTaxonomyDown = defaultdict(set)
    digraph_test = nx.DiGraph()
    with open(os.path.join(PATH, 'wikidata_2026_taxonomy.csv'), 'r') as topreader:
        for line in topreader:
            terms = line.strip().split(', ')
            if len(terms) >= 2:
                child, parent = terms[0], terms[1]
                child = child
                parent = parent
                wikiTaxonomyDown[parent].add(child)
                digraph_test.add_edge(parent, child)

    # first check the statistics
    stats = {
        "number_of_nodes": digraph_test.number_of_nodes(),
        "number_of_edges": digraph_test.number_of_edges(),
        "max_depth": max(nx.shortest_path_length(digraph_test, source='Q35120').values()),
        "weakly_connected": nx.is_weakly_connected(digraph_test),
        "directed_acyclic": nx.is_directed_acyclic_graph(digraph_test),
        "number_of_roots": len([node for node in digraph_test.nodes() if not list(digraph_test.predecessors(node))]),
        "number_of_leaves": len([node for node in digraph_test.nodes() if digraph_test.out_degree(node) == 0]),
        "number_of_internal_nodes": len([node for node in digraph_test.nodes() if digraph_test.out_degree(node) > 0]),
        "average_in_degree": sum(dict(digraph_test.in_degree()).values()) / digraph_test.number_of_nodes(),
    }
    for key, value in stats.items():
        print(f"{key}: {value}")
    print('===================================================')

    
    # first to deal with the loops
    root = 'Q35120' # entity
    cleanWikiTaxonomyDown=defaultdict(set)
    cleanWikiTaxonomyUp=defaultdict(set)
    topClasses=wikiTaxonomyDown.get(root, []) # set of top-classes
    cleanWikiTaxonomyDown[root]=topClasses.copy()
    for c in topClasses:
        cleanWikiTaxonomyUp[c].add(root)
    # Also adding the root class to the TaxonomyUp
    cleanWikiTaxonomyUp[root]=set()
    # wikidataTaxonomyLabels[root]='"entity"'
    # wikidataTaxonomyDescription[root]='"anything that can be considered, discussed, or observed"'
    cleaner=wikidataCleaner(cleanWikiTaxonomyDown, cleanWikiTaxonomyUp, wikiTaxonomyDown)
    for topClass in topClasses: # dfs traversal
        for subclass in wikiTaxonomyDown.get(topClass, []):
            cleaner.addSubClass(topClass, subclass)
    print("  Info: Loops removed:", cleaner.loopCounter)



    # # load cls with labels
    data_path = '../../data/data_2026/'
    df = pd.read_csv(os.path.join(data_path, 'wikidata_class_labels_full.csv'))
    # reduce redundancy
    df = df.drop_duplicates(subset=['item'])
    cls2label = df.set_index('item')['itemLabel'].to_dict()
    df = pd.read_csv(os.path.join(data_path, 'wikidata_class_descriptions_full.csv'))
    # reduce redundancy
    df = df.drop_duplicates(subset=['item'])
    cls2desc = df.set_index('item')['itemDesc'].to_dict()
    cls2label['Q35120'] = 'entity'
    cls2desc['Q35120'] = 'anything that can be considered, discussed, or observed'

    # load non_valid classes (instances of metaclasses and its subclasses)
    non_valid_classes = set()
    with open('../../data/data_2026/insts_of_metaclasses.txt', 'r') as f:
        for line in f:
            non_valid_classes.add(line.strip()[3:])

    # second: stop if we meet a class without label
    # statistics: number of classes without labels, number of classes in non_valid_classes
    print("Dealing with classes without labels or non_valid classes...")
    no_labels = set(digraph_test.nodes()) - set(cls2label.keys())
    non_valid = set(digraph_test.nodes()) & non_valid_classes
    print("----Number of classes without labels:", len(no_labels))
    print("----Number of classes in non_valid_classes:", len(non_valid))
    aftercleanWikiTaxonomyDown = defaultdict(set)
    root = 'Q35120' # entity
    topClasses = cleanWikiTaxonomyDown.get(root, []) # set of top-classes
    aftercleanWikiTaxonomyDown[root] = topClasses.copy()
    def addSubClass(superClass, subClass):
        """Adds the Wikidata classes to the wiki clean taxonomy"""
        if subClass not in cls2label:
            return
        if subClass in non_valid_classes:
            return
        # wikiTaxonomyUp[subClass].add(superClass)
        aftercleanWikiTaxonomyDown[superClass].add(subClass)
        # Avoid adding the subclasses again in case of double inheritance -> save time
        if subClass in aftercleanWikiTaxonomyDown:
            return
        for subClass2 in cleanWikiTaxonomyDown.get(subClass,[]):    
            addSubClass(subClass, subClass2)
    
    for topClass in topClasses: # DFS traversal
        for subclass in cleanWikiTaxonomyDown.get(topClass, []):
            addSubClass(topClass, subclass)
    print("  Info: Classes without labels:", len(set(aftercleanWikiTaxonomyDown.keys()) - set(cls2label.keys())), set(aftercleanWikiTaxonomyDown.keys()) - set(cls2label.keys()))
    digraph_test_new = nx.DiGraph(aftercleanWikiTaxonomyDown)
    stats = {
        "number_of_nodes": digraph_test_new.number_of_nodes(),
        "number_of_edges": digraph_test_new.number_of_edges(),
        "max_depth": max(nx.shortest_path_length(digraph_test_new, source='Q35120').values()),
        "weakly_connected": nx.is_weakly_connected(digraph_test_new),
        "directed_acyclic": nx.is_directed_acyclic_graph(digraph_test_new),
        "number_of_roots": len([node for node in digraph_test_new.nodes() if not list(digraph_test_new.predecessors(node))]),
    }
    for key, value in stats.items():
        print(f"{key}: {value}")
    print('===================================================')

    
    # Skip-connection (bypass) for classes without descriptions
    print("Dealing with classes without descriptions...")
    topgraph = nx.DiGraph(aftercleanWikiTaxonomyDown)
    cls_nodesc = set(topgraph.nodes) - set(cls2desc.keys())
    cls_nolabel = set(topgraph.nodes) - set(cls2label.keys())
    cls_nolabel_nodesc_union = cls_nolabel.union(cls_nodesc)
    for cls in tqdm(cls_nodesc, desc="Bypass classes without descriptions..."):
        if not topgraph.has_node(cls):
            continue
        # leaf nodes
        if topgraph.out_degree(cls) == 0:
            topgraph.remove_node(cls)
            continue
        # inner nodes
        children = list(topgraph.successors(cls))
        parents = list(topgraph.predecessors(cls))
        for pc in parents:
            for cc in children:
                topgraph.add_edge(pc, cc) # relink
        topgraph.remove_node(cls)
        # transitive reduction
        # print("  Info: Transitive reduction...,", cls)
        # topgraph = nx.transitive_reduction(topgraph)

    stats = {
        "number_of_nodes": topgraph.number_of_nodes(),
        "number_of_edges": topgraph.number_of_edges(),
        "max_depth": max(nx.shortest_path_length(topgraph, source='Q35120').values()),
        "weakly_connected": nx.is_weakly_connected(topgraph),
        "directed_acyclic": nx.is_directed_acyclic_graph(topgraph),
        "number_of_roots": len([node for node in topgraph.nodes() if not list(topgraph.predecessors(node))]),
    }
    for key, value in stats.items():
        print(f"{key}: {value}")
    print('===================================================')

    # remove top level classes with no subclasses
    print("Dealing with top level classes with no subclasses...")
    topclss = list(topgraph.successors('Q35120')) # entity
    cnt = 0
    for topcls in topclss:
        if topgraph.out_degree(topcls) == 0:
            topgraph.remove_node(topcls)
            cnt += 1
            print(f"  Info: Removed top level class: {topcls}")
    print("Number of top level classes with no subclasses:", cnt)

    # remove classes ScholarlyArticle and its subclasses
    tx_down = nx.to_dict_of_lists(topgraph)
    cls_discard = getDescendants(ScholarlyArticle, tx_down)
    for cls in tqdm(cls_discard, desc="Removing classes ScholarlyArticle and its subclasses..."):
        if not topgraph.has_node(cls):
            continue
        topgraph.remove_node(cls)
    print("Number of classes removed:", len(cls_discard))

    stats = {
        "number_of_nodes": topgraph.number_of_nodes(),
        "number_of_edges": topgraph.number_of_edges(),
        "max_depth": max(nx.shortest_path_length(topgraph, source='Q35120').values()),
        "weakly_connected": nx.is_weakly_connected(topgraph),
        "directed_acyclic": nx.is_directed_acyclic_graph(topgraph),
        "number_of_roots": len([node for node in topgraph.nodes() if not list(topgraph.predecessors(node))]),
    }
    for key, value in stats.items():
        print(f"{key}: {value}")
    print('===================================================')

    # remove non-informative classes
    print("Dealing with non-informative classes...")

    cls_inst_count = dict()
    with open('data/data_2026/cls_inst_count.txt', 'r') as f:
        for line in f:
            cls, count = line.strip().split('\t')
            cls_inst_count[cls] = int(count)

    N_nodes = topgraph.number_of_nodes()
    non_info_cls = find_non_informative_cls(topgraph, cls_inst_count)
    while len(non_info_cls) > 0:
        # remove non-informative classes recursively
        for node in tqdm(non_info_cls, desc='Filter non-informative classes'):
            parent = list(topgraph.predecessors(node))[0]
            child = list(topgraph.successors(node))[0]
            topgraph.remove_node(node)
            if not nx.has_path(topgraph, parent, child):
                topgraph.add_edge(parent, child)
        non_info_cls = find_non_informative_cls(topgraph, cls_inst_count)
    print("Number of classes removed:", N_nodes - topgraph.number_of_nodes())
    
    stats = {
        "number_of_nodes": topgraph.number_of_nodes(),
        "number_of_edges": topgraph.number_of_edges(),
        "max_depth": max(nx.shortest_path_length(topgraph, source='Q35120').values()),
        "weakly_connected": nx.is_weakly_connected(topgraph),
        "directed_acyclic": nx.is_directed_acyclic_graph(topgraph),
        "number_of_roots": len([node for node in topgraph.nodes() if not list(topgraph.predecessors(node))]),
    }
    for key, value in stats.items():
        print(f"{key}: {value}")
    print('===================================================')

    
    # remove bfo classes
    print("Dealing with BFO classes...")
    bfs_classes = set()
    bfs_names = dict()
    with open('data/data_2026/BFO_classes.csv', 'r') as f:
        for line in f:
            bfs_classes.add(line.strip().split(',')[0][3:])
            bfs_names[line.strip().split(',')[0][3:]] = line.strip().split(',')[1]
    print("Number of BFS classes:", len(bfs_classes))

    bfo_classes_in_graph = bfs_classes & set(topgraph.nodes())
    print("Number of BFS classes in graph:", len(bfo_classes_in_graph), [bfs_names[cls] for cls in bfo_classes_in_graph])

    graph = topgraph.copy()
    for cls in bfo_classes_in_graph:
        if cls == 'Q223557': # physical object
            continue
        if graph.has_node(cls):
            graph.remove_node(cls)

        # graph.add_edge('Q35120', 'Q27096213') # geoitem
        graph.add_edge('Q35120', 'Q214339') # role to entity
        graph.add_edge('Q35120', 'Q1190554') # occurance
        

        non_reachable_nodes = set()
        all_nodes = set(nx.descendants(graph, 'Q35120'))
        for node in graph.nodes:
            if node == 'Q35120':
                continue
            if node not in all_nodes:
                non_reachable_nodes.add(node)
                # graph.remove_node(node)

        for node in non_reachable_nodes:
            graph.remove_node(node)

    print("Number of classes removed after BFO:", topgraph.number_of_nodes() - len(graph.nodes()))
    stats = {
        "number_of_nodes": graph.number_of_nodes(),
        "number_of_edges": graph.number_of_edges(),
        "max_depth": max(nx.shortest_path_length(graph, source='Q35120').values()),
        "weakly_connected": nx.is_weakly_connected(graph),
        "directed_acyclic": nx.is_directed_acyclic_graph(graph),
        "number_of_roots": len([node for node in graph.nodes() if not list(graph.predecessors(node))]),
    }
    for key, value in stats.items():
        print(f"{key}: {value}")
    print('===================================================')


    # remove classes without cumulative instances
    topgraph = graph.copy()
    print("Dealing with classes without cumulative instances...")
    cls_no_inst = set()
    set_cls = set(topgraph.nodes())
    tx_down = nx.to_dict_of_lists(digraph_test)
    for cls in tqdm(set_cls, desc="Removing classes without cumulative instances..."):
        if not topgraph.has_node(cls):
            continue
        if cls in cls_inst_count:
            continue
        if cumulative_stats_for_class(cls, cls_inst_count, tx_down) < 1:
            cls_no_inst.add(cls)
            topgraph.remove_node(cls)
    print("Number of classes removed:", len(cls_no_inst))

    # stats
    stats = {
        "number_of_nodes": topgraph.number_of_nodes(),
        "number_of_edges": topgraph.number_of_edges(),
        "max_depth": max(nx.shortest_path_length(topgraph, source='Q35120').values()),
        "weakly_connected": nx.is_weakly_connected(topgraph),
        "directed_acyclic": nx.is_directed_acyclic_graph(topgraph),
        "number_of_roots": len([node for node in topgraph.nodes() if not list(topgraph.predecessors(node))]),
    }
    for key, value in stats.items():
        print(f"{key}: {value}")
    print('===================================================')
    

    print("Remove rare classes: without Wikipedia page...")
    wikipedia_loc = '../../data/data_2026/wikidata/wikipedia'
    wikc_noisy = remove_rare_classes(topgraph, wikipedia_loc, root='Q35120', depth=3)
    stats = {
        "number_of_nodes": wikc_noisy.number_of_nodes(),
        "number_of_edges": wikc_noisy.number_of_edges(),
        "max_depth": max(nx.shortest_path_length(wikc_noisy, source='Q35120').values()),
        "weakly_connected": nx.is_weakly_connected(wikc_noisy),
        "directed_acyclic": nx.is_directed_acyclic_graph(wikc_noisy),
        "number_of_roots": len([node for node in wikc_noisy.nodes() if not list(wikc_noisy.predecessors(node))]),
    }
    for key, value in stats.items():
        print(f"{key}: {value}")
    print('===================================================')


    # see current top level classes
    top_level_classes = list(topgraph.successors('Q35120'))
    print("Top-level classes:")
    for cls in top_level_classes:
        print(f"{cls}: {cls2label[cls]}")


    # save the taxonomy
    with open(os.path.join(PATH, 'noisy_wikidata_extracted.tsv'), 'w') as f:
        for edge in wikc_noisy.edges():
            parent, child = edge
            f.write(f"{child}\trdfs:subClassOf\t{parent}\t.\n")

    
