from collections import defaultdict
from scipy.sparse import csr_matrix
from tqdm import tqdm
import networkx as nx
import graph_utils
import torch
import pickle
import json
import os

class TaxonCleaner:
    def __init__(self, data_path, model='llama8b'):
        # self.graph = graph
        self.root = "wd:Q35120" # entity
        self.model_name = model
        self.cls2label = None
        self.cls_inst_stats = dict() # direct instance count for each class
        self.irrel_pred = None # list of tuples (parent, child, confidence)
        self.equiv_pred = None
        self.reverse_pred = None
        self.subsume_pred = None
        self.rewire_links = None # dict of tuples (child, parent): list of hops for that tuple
        self.data_path = data_path
        self.mapping = dict() # mapping from Wikidata to WiKC
        self.edges_del = set() # edges deleted during cleaning: parent -> child

        # self.taxonomy_down_to_top = None # child -> parent
        # self.taxonomy_top_to_down = None # parent -> child
        self.wiki_dag = None # directed acyclic graph of the taxonomy
        self._init_taxonomy()
        self._calc_cls_inst_stats()


    def _init_taxonomy(self):
        """ 
        Initializes the taxonomy from the data path.
        """
        # self.taxonomy_down_to_top = defaultdict(set)
        taxonomy_top_to_down = defaultdict(set)
        with open(os.path.join(self.data_path, 'noisy_wikidata_2026.tsv'), 'r') as topreader: # TBC
        # with open(os.path.join(self.data_path, 'noisy_WiKC.tsv'), 'r') as topreader:
            for line in topreader:
                triple = line.strip().split('\t')
                if len(triple) > 3:
                    child, parent = triple[0], triple[2]
                    if not child.startswith('wd:'):
                        child = 'wd:' + child
                    if not parent.startswith('wd:'):
                        parent = 'wd:' + parent
                    # self.taxonomy_down_to_top[child].add(parent)
                    taxonomy_top_to_down[parent].add(child)
        self.wiki_dag = nx.DiGraph(taxonomy_top_to_down)


    def _calc_cls_inst_stats(self):
        """ 
        Calculates the statistics of direct instances for each class. 
        For each leaf node, it counts the cumulative instances.
        """
        oriwikiDown = defaultdict(set)
        with open(os.path.join(self.data_path, 'wiki_2026_taxonomy_full_v2.txt'), 'r') as file:
            for line in file:
                terms = line.strip().split(',')
                if len(terms) > 1:
                    child, parent = terms[0], terms[1]
                    if not child.startswith('wd:'):
                        child = 'wd:' + child
                    if not parent.startswith('wd:'):
                        parent = 'wd:' + parent
                    oriwikiDown[parent].add(child)
        cls_inst_count = dict()
        with open(os.path.join(self.data_path, 'cls_inst_count_2026_filter_transitivity.txt'), 'r') as file:
            for line in file:
                terms = line.strip().split('\t')
                if len(terms) > 1:
                    cls, count = terms[0], terms[1]
                    if not cls.startswith('wd:'):
                        cls = 'wd:' + cls
                    cls_inst_count[cls] = int(count)
        
        # init the statistics of each class in noisy WiKC
        for node in self.wiki_dag.nodes:
            if self.wiki_dag.out_degree(node) == 0:
                # leaf node, count the cumulative instances
                self.cls_inst_stats[node] = graph_utils.cumulative_stats_for_class(node, cls_inst_count, oriwikiDown)
            elif node in cls_inst_count: # non-leaf nodes
                self.cls_inst_stats[node] = cls_inst_count[node]
        # check if all leaves in wiki_dag are in cls_inst_stats
        cnt = 0
        for node in self.wiki_dag.nodes:
            if self.wiki_dag.out_degree(node) == 0:
                if node not in self.cls_inst_stats:
                    cnt += 1
                    # raise ValueError(f"Leaf node {node} not in cls_inst_stats.")
                    self.cls_inst_stats[node] = 1
        print(f"    {cnt} leaves not in cls_inst_stats, set to 1.")


    def deduce_predictions(self, res_file, threshold=0.5):
        """ Deduces predictions from the LLM results file.
        Args:
            res_file (str): Path to the LLM results file.
            threshold (float): Confidence threshold for considering a prediction valid.
        Returns:
            tuple: Four lists containing tuples of (parent, child, confidence) for each category:
                - irrel: Irrelevant relationships
                - equiv: Equivalent relationships
                - reverse: Reverse relationships
                - subsume: Subsuming relationships  
        """
        self.irrel_pred, self.equiv_pred, self.reverse_pred, self.subsume_pred = {}, {}, {}, {}
        with open(res_file, 'r') as f:
            llm_res = json.load(f)
        # Process llm_res to deduce predictions
        for res in llm_res:
            parent, child = res['id'].split('_')
            if res['plabel'].lower() == res['clabel'].lower():
                # if parent and child share the same label, consider it as equivalent -> exact match
                self.equiv_pred[tuple([parent, child])] = max(res['confidence'], res['confidence_inverse'])
                continue
            # deduce relationships based on LLM's answers
            if res['answer'].lower() == 'true' and res['answer_inverse'].lower() == 'false':
                self.subsume_pred[tuple([parent, child])] = max(res['confidence'], res['confidence_inverse'])
            elif res['answer'].lower() == 'true' and res['answer_inverse'].lower() == 'true':
                conf = max(res['confidence'], res['confidence_inverse'])
                if conf >= threshold:
                    self.equiv_pred[tuple([parent, child])] = conf
                else:
                    self.subsume_pred[tuple([parent, child])] = conf
            elif res['answer'].lower() == 'false' and res['answer_inverse'].lower() == 'true':
                conf = max(res['confidence'], res['confidence_inverse'])
                if conf >= threshold:
                    self.reverse_pred[tuple([parent, child])] = conf
                else:
                    self.subsume_pred[tuple([parent, child])] = conf
            elif res['answer'].lower() == 'false' and res['answer_inverse'].lower() == 'false':
                conf = max(res['confidence'], res['confidence_inverse'])
                if conf >= threshold:
                    self.irrel_pred[tuple([parent, child])] = conf
                else:
                    self.subsume_pred[tuple([parent, child])] = conf
            else: # LLM failed to answer, keep as it is
                self.subsume_pred[tuple([parent, child])] = max(res['confidence'], res['confidence_inverse'])
                print(f"LLM failed to answer for {parent} -> {child}, {res['answer']} / {res['answer_inverse']}")

        # show statistics
        total = len(self.irrel_pred) + len(self.equiv_pred) + len(self.reverse_pred) + len(self.subsume_pred)
        print(f"Total predictions: {total}")
        print(f"Irrelevant: {len(self.irrel_pred)}, Percentage: {len(self.irrel_pred) / total * 100:.2f}%")
        print(f"Equivalent: {len(self.equiv_pred)}, Percentage: {len(self.equiv_pred) / total * 100:.2f}%")
        print(f"Reverse: {len(self.reverse_pred)}, Percentage: {len(self.reverse_pred) / total * 100:.2f}%")
        print(f"Subsuming: {len(self.subsume_pred)}, Percentage: {len(self.subsume_pred) / total * 100:.2f}%")

        # check if all edges in wiki_dag are in the predictions
        unpredicted_edges = set()
        for edge in self.wiki_dag.edges():
            if edge not in self.irrel_pred and edge not in self.equiv_pred and edge not in self.reverse_pred and edge not in self.subsume_pred:
                unpredicted_edges.add(edge)
        if len(unpredicted_edges) > 0:
            with open(f"unpredicted_edges_{self.model_name}.txt", "w") as f:
                for edge in unpredicted_edges:
                    parent, child = edge
                    f.write(f"{child}\t{parent}\n")
            print(f"Unpredicted edges: {len(unpredicted_edges)}")
            raise ValueError(f"Unpredicted edges: {len(unpredicted_edges)}")
    
    def store_predictions(self, loc):
        if not os.path.exists(loc):
            raise FileNotFoundError(f"Directory does not exist: {loc}")
        with open(os.path.join(loc, f'{self.model_name}_predictions.txt'), 'w') as f:
            for edge, conf in self.irrel_pred.items():
                parent, child = edge
                f.write(f"{child}\t{parent}\t[IRRELEVANT]\t{conf}\n") # child -> parent format edge
            for edge, conf in self.equiv_pred.items():
                parent, child = edge
                f.write(f"{child}\t{parent}\t[EQUIVALENT]\t{conf}\n")
            for edge, conf in self.reverse_pred.items():
                parent, child = edge
                f.write(f"{child}\t{parent}\t[REVERSE]\t{conf}\n")
            for edge, conf in self.subsume_pred.items():
                parent, child = edge
                f.write(f"{child}\t{parent}\t[SUBSUME]\t{conf}\n")

    
    def store_majority_predictions(self, loc, model_names=[]):
        if not os.path.exists(loc):
            raise FileNotFoundError(f"Directory does not exist: {loc}")
        # load all model predictions, then do majority voting
        model_predictions = dict()
        for model_name in model_names:
            model_predictions[model_name] = dict()
            with open(os.path.join(loc, model_name, f'{model_name}_predictions.txt'), 'r') as f:
                for line in f:
                    child, parent, pred_, conf = line.strip().split('\t')
                    model_predictions[model_name][tuple([parent, child])] = (pred_, float(conf))
        # do majority voting
        with open(os.path.join(loc, f'majority_predictions.txt'), 'w') as f:
            for edge in self.wiki_dag.edges(): # parent -> child format edge
                pred_counts = dict()
                for model_name in model_names:
                    if edge not in model_predictions[model_name]:
                        raise ValueError(f"Edge {edge} not found in {model_name} predictions.")
                    pred_, conf_ = model_predictions[model_name][edge]
                    pred_counts[pred_] = pred_counts.get(pred_, 0) + 1
                # check if max_count is unique
                max_count = max(pred_counts.values())
                valid_pred = True
                if list(pred_counts.values()).count(max_count) > 1:
                    valid_pred = False
                if valid_pred:
                    max_count = max(pred_counts.values())
                    pred_final = [pred for pred, count in pred_counts.items() if count == max_count][0]
                    f.write(f"{edge[1]}_{edge[0]}\t{pred_final}\t{valid_pred}\n") # child -> parent format edge
                else:
                    f.write(f"{edge[1]}_{edge[0]}\t{pred_counts}\t{valid_pred}\n")
                

    
    def load_majority_predictions(self, loc):
        if not os.path.exists(loc):
            raise FileNotFoundError(f"Directory does not exist: {loc}")
        self.irrel_pred, self.equiv_pred, self.reverse_pred, self.subsume_pred = {}, {}, {}, {}
        with open(os.path.join(loc, f'majority_predictions.txt'), 'r') as f:
            for line in f:
                item_id, pred_final, valid_pred = line.strip().split('\t')
                child, parent = item_id.split('_')
                if valid_pred == 'False':
                    pred_final = '[SUBSUME]'

                # classify
                if pred_final == '[IRRELEVANT]':
                    self.irrel_pred[tuple([parent, child])] = 1.0 # TBC
                elif pred_final == '[EQUIVALENT]':
                    self.equiv_pred[tuple([parent, child])] = 1.0
                elif pred_final == '[REVERSE]':
                    self.reverse_pred[tuple([parent, child])] = 1.0
                elif pred_final == '[SUBSUME]':
                    self.subsume_pred[tuple([parent, child])] = 1.0
                else:
                    raise ValueError(f"Invalid prediction: {pred_final}, {valid_pred}")
        # statistics
        total = len(self.irrel_pred) + len(self.equiv_pred) + len(self.reverse_pred) + len(self.subsume_pred)
        print(f"Total predictions: {total}")
        print(f"Irrelevant: {len(self.irrel_pred)}, Percentage: {len(self.irrel_pred) / total * 100:.2f}%")
        print(f"Equivalent: {len(self.equiv_pred)}, Percentage: {len(self.equiv_pred) / total * 100:.2f}%")
        print(f"Reverse: {len(self.reverse_pred)}, Percentage: {len(self.reverse_pred) / total * 100:.2f}%")
        print(f"Subsuming: {len(self.subsume_pred)}, Percentage: {len(self.subsume_pred) / total * 100:.2f}%")
        # check if all edges in wiki_dag are in the predictions
        unpredicted_edges = set()
        for edge in self.wiki_dag.edges():
            if edge not in self.irrel_pred and edge not in self.equiv_pred and edge not in self.reverse_pred and edge not in self.subsume_pred:
                unpredicted_edges.add(edge)
        if len(unpredicted_edges) > 0:
            raise ValueError(f"Unpredicted edges: {len(unpredicted_edges)}")         
    
    
    def get_rechecked_links(self):
        """ Returns the links need to be rechecked based on the predictions.
        Returns:
            list: List of tuples (child, parent) for rechecked links.
        """
        check_links = []
        rewire_links = []
        # resolve situations
        for edge, conf in self.reverse_pred.items():
            parent, child = edge
            if self.wiki_dag.has_edge(parent, child) and self.wiki_dag.in_degree(child) <= 1:
                # check if other children of 'parent' are subclasses of 'child'
                children = set(self.wiki_dag.successors(parent)) - set([child])
                for c in children:
                    check_links.append((c, child)) # child -> parent format edge

        # equivalent situations
        for edge, conf in self.equiv_pred.items():
            parent, child = edge
            if parent == self.root:
                continue

            # check if other parents of 'child' are superclasses of 'parent'
            for p in self.wiki_dag.predecessors(child):
                # avoid duplicates and contradiction
                if p != parent and not (nx.has_path(self.wiki_dag, p, parent) or nx.has_path(self.wiki_dag, parent, p)):
                    rewire_links.append((parent, p)) # child -> parent format edge

        return check_links, rewire_links

    def reduce(self):
        """ Applies transitive reduction on the directed acyclic graph wiki_dag. """
        self.wiki_dag = nx.transitive_reduction(self.wiki_dag)

    def cut(self, simi_matrix=None, node2id=None):
        """ 
        Applies the cut strategy to remove irrelevant edges and small disconnected subtrees.
        """
        bfs_edges = graph_utils.bfs_edges_by_level(self.wiki_dag, self.root)
        bfs_edges = graph_utils.reorder_edges_by_similarity(self.wiki_dag, bfs_edges, simi_matrix, node2id)
        for parent, child in tqdm(bfs_edges, desc='Cut Strategy'):
            # Skip if the edge does not exist in the wiki_dag
            if not self.wiki_dag.has_edge(parent, child):
                continue
            # Skip if the edge is not in the irrelevant predictions
            if tuple([parent, child]) not in self.irrel_pred:
                continue
            if self.wiki_dag.in_degree(child) > 1:
                # keep connected to the root, with multiple parents
                self.wiki_dag.remove_edge(parent, child)
                self.edges_del.add((parent, child))
            else:
                # only one parent, remove the disconnected subtree that has <= 3 nodes
                iso_nodes = [child]
                cut_subtree = True
                self.wiki_dag.remove_edge(parent, child) # temporarily remove the edge
                for layer in nx.bfs_layers(self.wiki_dag, child):
                    # skip the first layer (the child itself)
                    if layer == iso_nodes:
                        continue
                    cur_layer_connected = True
                    for node in layer:
                        if not nx.has_path(self.wiki_dag, self.root, node):
                            # not reachable from root
                            iso_nodes.append(node)
                            cur_layer_connected = False
                        if len(iso_nodes) > 3:
                            cut_subtree = False
                            break
                    # check if the current layer is fully connected to the root
                    if cur_layer_connected:
                        break
                    # check if still cutable
                    if not cut_subtree:
                        break
                
                if cut_subtree:
                    # remove the subtree
                    self.wiki_dag.remove_nodes_from(iso_nodes)
                    # update the cls_inst_stats, streamingly remove the entries
                    for node_ in iso_nodes:
                        self.cls_inst_stats.pop(node_, None)
                    self.edges_del.add((parent, child))
                else:
                    self.wiki_dag.add_edge(parent, child) # add back the edge


    def resolve(self, simi_matrix=None, node2id=None):
        """
        Applies the resolve strategy to fix reverse relationships by cutting or merging nodes.
        """
        bfs_edges = graph_utils.bfs_edges_by_level(self.wiki_dag, self.root) # degree has been changed since delete steps
        # reorder edges by similarity
        bfs_edges = graph_utils.reorder_edges_by_similarity(self.wiki_dag, bfs_edges, simi_matrix, node2id)
        for parent, child in tqdm(bfs_edges, desc='Resolve Strategy'):
            # Skip first-level classes: Don't want too complex at high level
            if parent == self.root:
                continue
            # Skip if the edge does not exist in the wiki_dag
            if not self.wiki_dag.has_edge(parent, child):
                continue
            # Skip if the edge is not in the reverse predictions
            if tuple([parent, child]) not in self.reverse_pred:
                continue
            # Cut or merge based on the connectivity
            if self.wiki_dag.in_degree(child) > 1:
                self.wiki_dag.remove_edge(parent, child)
                self.edges_del.add((parent, child))
            else: # merge
                for sc in self.wiki_dag.successors(child):
                    self.wiki_dag.add_edge(parent, sc)
                self.wiki_dag.remove_node(child)
                # save the mapping
                self.mapping[child] = parent
                # update the stats
                self.cls_inst_stats[parent] = self.cls_inst_stats.get(parent, 0) + self.cls_inst_stats.get(child, 0)
                self.cls_inst_stats.pop(child, None)
    
    
    def get_reiwre_links_new(self, bfs_edges):
        self.rewire_links = dict()
        for parent, child in bfs_edges:
            # Skip merge to root classes
            if parent == self.root:
                continue
            # Skip if the edge is not in the equivalent predictions
            if tuple([parent, child]) not in self.equiv_pred:
                continue
            # Check edge existence
            if not self.wiki_dag.has_edge(parent, child):
                continue

            # First get rewire links and their hops
            for p in self.wiki_dag.predecessors(child):
                # avoid duplicates and contradiction
                if p == parent:
                    continue
                if nx.has_path(self.wiki_dag, parent, p):
                    raise ValueError(f"Path found between {parent} and {p}, which will cause cycle after merging.")
                if p != parent and not nx.has_path(self.wiki_dag, p, parent):
                    if (parent, p) not in self.rewire_links:
                        self.rewire_links[(parent, p)] = []
                    # one more step: add multi-hop check (one-hop)
                    for p_ in self.wiki_dag.predecessors(p):
                        if nx.has_path(self.wiki_dag, parent, p_):
                            raise ValueError(f"Path found between {parent} and {p_}, which denotes cycles in current graph.")
                        if not nx.has_path(self.wiki_dag, p_, parent):
                            self.rewire_links[(parent, p)].append((parent, p_)) # child -> parent format edge
    
    
    def exist_prediction_for_all_reiwred_links(self, rewire_file_loc, simi_matrix=None, node2id=None):
        if not os.path.exists(rewire_file_loc):
            raise FileNotFoundError(f"Rewire results file not found:{rewire_file_loc}")
        with open(rewire_file_loc, 'r') as f:
            rewire_res = json.load(f)
            llm_rewire_res = dict()
            for res in rewire_res:
                parent_, child_ = res['id'].split('_')
                llm_rewire_res[(child_, parent_)] = res
        bfs_edges = graph_utils.bfs_edges_by_level(self.wiki_dag, self.root)
        # sorted by similarity in descending order
        bfs_edges = graph_utils.reorder_edges_by_similarity(self.wiki_dag, bfs_edges, simi_matrix, node2id, reverse=True)
        self.get_reiwre_links_new(bfs_edges)
        # check if all edges are in the llm_rewire_res
        unpredicted_edges = set()
        for edge, hop in self.rewire_links.items():
            if edge not in llm_rewire_res:
                unpredicted_edges.add(edge)
            for hop_edge in hop:
                if hop_edge not in llm_rewire_res:
                    unpredicted_edges.add(hop_edge) # child -> parent format edge
        return unpredicted_edges


    def check_valid_merge_edges(self, rewire_file_loc, threshold_rewire=0.5, save_loc=None, simi_matrix=None, node2id=None):
        bfs_edges = graph_utils.bfs_edges_by_level(self.wiki_dag, self.root)
        # sorted by similarity in descending order
        bfs_edges = graph_utils.reorder_edges_by_similarity(self.wiki_dag, bfs_edges, simi_matrix, node2id, reverse=True)
        if not os.path.exists(rewire_file_loc):
            # raise FileNotFoundError(f"Rewire results file not found: {reiwre_file_loc}")
            # print(f"    Warning: Rewire results file not found: {rewire_file_loc}")
            self.get_reiwre_links_new(bfs_edges)
            if save_loc is None:
                raise ValueError("save_loc has to be provided if rewire_file_loc is not found.")
            self.store_rewire_links(save_loc) # save_loc has to be provided
            print(f"    Rewired links saved to {save_loc}")
            raise FileNotFoundError(f"Rewire results file not found: {rewire_file_loc}, has to first calculate the rewire links.")
        else:
            with open(rewire_file_loc, 'r') as f:
                rewire_res = json.load(f)
            llm_rewire_res = dict()
            for res in rewire_res:
                parent_, child_ = res['id'].split('_')
                llm_rewire_res[(child_, parent_)] = res
        
        # get valid merge edges
        valid_merge_edges = list() # ordered list
        cnt = 0
        for parent, child in tqdm(bfs_edges, desc='Get Valid Merge Edges'):
            local_rewire_links = dict()
            # Skip merge to root classes
            if parent == self.root:
                continue
            # Skip if the edge is not in the equivalent predictions
            if tuple([parent, child]) not in self.equiv_pred:
                continue
            # Merge process: no continuous merging
            if not self.wiki_dag.has_edge(parent, child):
                continue

            # count the number of nodes that have more than one parent
            num_parents = len(set(self.wiki_dag.predecessors(child)))
            if num_parents > 1:
                cnt += 1

            for p in self.wiki_dag.predecessors(child):
                # avoid duplicates and contradiction
                if p == parent:
                    continue
                if nx.has_path(self.wiki_dag, parent, p):
                    raise ValueError(f"Path found between {parent} and {p}, which will cause cycle after merging.")
                if p != parent and not nx.has_path(self.wiki_dag, p, parent):
                    if (parent, p) not in local_rewire_links:
                        local_rewire_links[(parent, p)] = []
                    # one more step: add multi-hop check
                    for p_ in self.wiki_dag.predecessors(p):
                        if nx.has_path(self.wiki_dag, parent, p_):
                            raise ValueError(f"Path found between {parent} and {p_}, which denotes cycles in current graph.")
                        if not nx.has_path(self.wiki_dag, p_, parent):
                            local_rewire_links[(parent, p)].append((parent, p_)) # child -> parent format edge
            
            # check if the rewire links are valid
            valid = True
            while local_rewire_links:
                # get one item from local_rewire_links and delete it
                edge, hop = local_rewire_links.popitem()
                if edge not in llm_rewire_res:
                    print("Edge not found in LLM results:", child, parent, edge, hop)
                    continue
                # check subsumption
                ans = llm_rewire_res[edge]['answer']
                conf = llm_rewire_res[edge]['confidence']
                inv_ans = llm_rewire_res[edge]['answer_inverse']
                inv_conf = llm_rewire_res[edge]['confidence_inverse']
                if ans.lower() == 'true' and inv_ans.lower() == 'false' and min(conf, inv_conf) >= threshold_rewire:
                    # multi-hop check
                    for hop_edge in hop:
                        if hop_edge not in llm_rewire_res:
                            print("Hop edge not found in LLM results:", child, parent, edge, hop, hop_edge)
                            continue
                        ans_hop = llm_rewire_res[hop_edge]['answer']
                        inv_ans_hop = llm_rewire_res[hop_edge]['answer_inverse']
                        if ans_hop.lower() == 'true' and inv_ans_hop.lower() == 'false' and \
                            min(llm_rewire_res[hop_edge]['confidence'], llm_rewire_res[hop_edge]['confidence_inverse']) >= threshold_rewire:
                            continue
                        valid = False
                        break
                    # continue
                else:
                    valid = False
                    
                if not valid:
                    break
            
            # merge if it is valid
            if valid:
                valid_merge_edges.append((parent, child))
        print(f"    Found {len(valid_merge_edges)} valid merge edges.")
        print(f"    {cnt} nodes have more than one parent.")
        return valid_merge_edges


    def merge_new(self, valid_merge_edges: list):
        merged_cls = set()
        cnt = 0
        for parent, child in tqdm(valid_merge_edges, desc='Merge & Rewire Strategy'):
            # Skip merge to root classes
            if parent == self.root:
                continue
            # Skip if the edge is not in the equivalent predictions
            if tuple([parent, child]) not in self.equiv_pred:
                continue
            # Merge process: no continuous merging
            if not self.wiki_dag.has_edge(parent, child):
                continue
            if child in merged_cls:
                continue

            # Reconnect the successors 
            for sc in self.wiki_dag.successors(child):
                self.wiki_dag.add_edge(parent, sc)

            for p in self.wiki_dag.predecessors(child):
                if p != parent and nx.has_path(self.wiki_dag, parent, p):
                    raise ValueError(f"Path found between {parent} and {p}, which causes cycle after merging.")
                if p != parent:
                    self.wiki_dag.add_edge(p, parent)
            self.wiki_dag.remove_node(child)
            self.mapping[child] = parent
            merged_cls.add(parent)
            self.reduce() # cost time
            # update stats
            self.cls_inst_stats[parent] = self.cls_inst_stats.get(parent, 0) + self.cls_inst_stats.get(child, 0)
            self.cls_inst_stats.pop(child, None)
            cnt += 1
        print(f"    Merged {cnt} edges.")
    
    def merge_new_4(self, valid_merge_edges: list):
        merged_cls = set()
        for node in self.mapping.keys():
            merged_cls.add(self.mapping[node]) # merged classes at resolve step
        cnt = 0
        for parent, child in tqdm(valid_merge_edges, desc='Merge & Rewire Strategy'):
            # Skip merge to root classes
            if parent == self.root:
                continue
            # Skip if the edge is not in the equivalent predictions
            if tuple([parent, child]) not in self.equiv_pred:
                continue
            # Merge process: no continuous merging
            if not self.wiki_dag.has_edge(parent, child):
                continue
            if child in merged_cls:
                continue
            
            if parent in merged_cls:
                continue

            # Reconnect the successors 
            for sc in self.wiki_dag.successors(child):
                self.wiki_dag.add_edge(parent, sc)

            for p in self.wiki_dag.predecessors(child):
                if p != parent and nx.has_path(self.wiki_dag, parent, p):
                    raise ValueError(f"Path found between {parent} and {p}, which causes cycle after merging.")
                if p != parent:
                    self.wiki_dag.add_edge(p, parent)
            self.wiki_dag.remove_node(child)
            self.mapping[child] = parent
            merged_cls.add(parent)
            self.reduce() # cost time
            # update stats
            self.cls_inst_stats[parent] = self.cls_inst_stats.get(parent, 0) + self.cls_inst_stats.get(child, 0)
            self.cls_inst_stats.pop(child, None)
            cnt += 1
        print(f"    Merged {cnt} edges.")

    
    def merge_new_2(self, valid_merge_edges: list):
        merged_cls = set()
        cnt = 0
        for parent, child in tqdm(valid_merge_edges, desc='Merge & Rewire Strategy'):
            # Skip merge to root classes
            if parent == self.root:
                continue
            # Skip if the edge is not in the equivalent predictions
            if tuple([parent, child]) not in self.equiv_pred:
                continue
            # Merge process: no continuous merging
            if not self.wiki_dag.has_edge(parent, child):
                continue
            if child in merged_cls:
                continue

            # Reconnect the successors 
            for sc in self.wiki_dag.successors(child):
                self.wiki_dag.add_edge(parent, sc)

            for p in self.wiki_dag.predecessors(child):
                if p != parent and nx.has_path(self.wiki_dag, parent, p):
                    raise ValueError(f"Path found between {parent} and {p}, which causes cycle after merging.")
                if p != parent:
                    # 04-23 new added here, add link between p and all successors of child
                    for sc in self.wiki_dag.successors(child):
                        self.wiki_dag.add_edge(p, sc)
            self.wiki_dag.remove_node(child)
            self.mapping[child] = parent
            merged_cls.add(parent)
            self.reduce() # cost time
            # update stats
            self.cls_inst_stats[parent] = self.cls_inst_stats.get(parent, 0) + self.cls_inst_stats.get(child, 0)
            self.cls_inst_stats.pop(child, None)
            cnt += 1
        print(f"    Merged {cnt} edges.")

    
    def merge_new_3(self, valid_merge_edges: list, simi_matrix=None, node2id=None, path_file_loc=None):
        merged_cls = set()
        cnt = 0

        bfs_edges = graph_utils.bfs_edges_by_level(self.wiki_dag, self.root)
        # sorted by similarity in descending order
        bfs_edges = graph_utils.reorder_edges_by_similarity(self.wiki_dag, bfs_edges, simi_matrix, node2id, reverse=True)
        merged_edges = set()
        for parent, child in tqdm(bfs_edges, desc='Merge & Rewire Strategy'):
            # Skip merge to root classes
            if parent == self.root:
                continue
            # Skip if the edge is not in the equivalent predictions
            if tuple([parent, child]) not in self.equiv_pred:
                continue
            # Merge process: no continuous merging
            if not self.wiki_dag.has_edge(parent, child):
                continue
            if child in merged_cls:
                continue

            # Reconnect the successors 
            for sc in self.wiki_dag.successors(child):
                self.wiki_dag.add_edge(parent, sc)

            for p in self.wiki_dag.predecessors(child):
                if p != parent and nx.has_path(self.wiki_dag, parent, p):
                    raise ValueError(f"Path found between {parent} and {p}, which causes cycle after merging.")
                # Chenge here
                if p != parent:
                    for sc in self.wiki_dag.successors(child):
                        self.wiki_dag.add_edge(p, sc)

            self.wiki_dag.remove_node(child)
            self.mapping[child] = parent
            merged_cls.add(parent)
            self.reduce() # cost time
            # update stats
            self.cls_inst_stats[parent] = self.cls_inst_stats.get(parent, 0) + self.cls_inst_stats.get(child, 0)
            self.cls_inst_stats.pop(child, None)
            cnt += 1
            merged_edges.add((parent, child))
        print(f"    Merged {cnt} edges.")
        # save additional merged edges
        additional_merged_edges = set(merged_edges) - set(valid_merge_edges)
        with open(path_file_loc, 'w') as f:
            for edge in additional_merged_edges:
                parent, child = edge
                f.write(f"{child}\t{parent}\n")
        
    


    def get_rewire_links(self, local_rewire_links):
        """ 
        Returns the links need to be rechecked based on the predictions.
        Returns:
            dict: tuples of (child, parent): upper hops for that tuple
        """
        self.rewire_links = dict()
        for child, parent in local_rewire_links:
            if not self.wiki_dag.has_node(child) or not self.wiki_dag.has_node(parent):
                continue
            reverse_pred = (parent, child)
            if reverse_pred in local_rewire_links:
                continue
            if (child, parent) in self.rewire_links:
                raise ValueError("Duplicate rewire link found!")
            
            # add multi-hop check
            self.rewire_links[(child, parent)] = []
            # for c in cleaner.wiki_dag.successors(child):
            #     if not nx.has_path(cleaner.wiki_dag, c, parent) and not nx.has_path(cleaner.wiki_dag, parent, c):
            #         # no path between c and parent
            #         self.rewire_links[(child, parent)].append((c, parent)) # child -> parent format edge
            for p in self.wiki_dag.predecessors(parent):
                if not nx.has_path(self.wiki_dag, child, p) and not nx.has_path(self.wiki_dag, p, child):
                    self.rewire_links[(child, parent)].append((child, p)) # child -> parent format edge
    
    def store_rewire_links(self, loc):
        if not os.path.exists(loc):
            raise FileNotFoundError(f"Directory does not exist: {loc}")
        # store the rewire links
        total_rewire_edges = set()
        for k, v in self.rewire_links.items():
            total_rewire_edges.add(k) # include edge itself
            for edge in v:
                total_rewire_edges.add(edge)
        with open(os.path.join(loc, f'{self.model_name}_rewire_links.txt'), 'w') as f:
            for edge in total_rewire_edges:
                child, parent = edge
                f.write(f"{child}\t{parent}\n")

    
    def rewire(self, file_loc, threshold_rewire=0.5):
        """ 
        Applies the rewire strategy to add potential edges based on LLM predictions.
        Args:
            file_loc (str): Path to the LLM results file for rewire links.
            threshold_rewire (float): Confidence threshold for considering a rewire prediction valid.
        """
        if not os.path.exists(file_loc):
            raise FileNotFoundError(f"Rewire results file not found: {file_loc}")
        with open(file_loc, 'r') as f:
            rewire_res = json.load(f)
        llm_rewire_res = dict()
        for res in rewire_res:
            parent, child = res['id'].split('_')
            llm_rewire_res[(child, parent)] = res
        
        # apply rewire strategy
        for edge, hop in self.rewire_links.items():
            child, parent = edge
            if edge not in llm_rewire_res:
                print("Edge not found in LLM results:", edge)
                continue
            # check subsumption
            ans = llm_rewire_res[edge]['answer']
            conf = llm_rewire_res[edge]['confidence']
            inv_ans = llm_rewire_res[edge]['answer_inverse']
            inv_conf = llm_rewire_res[edge]['confidence_inverse']
            if ans.lower() == 'true' and inv_ans.lower() == 'false' and min(conf, inv_conf) >= threshold_rewire:
                if self.wiki_dag.has_edge(parent, child):
                    print("Edge already exists in the graph:", edge)
                    continue
                # check hops
                consistency = True
                for hop_edge in hop:
                    ans_hop = llm_rewire_res[hop_edge]['answer']
                    inv_ans_hop = llm_rewire_res[hop_edge]['answer_inverse']
                    if ans_hop.lower() == 'true' and inv_ans_hop.lower() == 'false' and \
                        min(llm_rewire_res[hop_edge]['confidence'], llm_rewire_res[hop_edge]['confidence_inverse']) >= threshold_rewire:
                        continue
                    consistency = False
                    break
                if consistency:
                    self.wiki_dag.add_edge(parent, child)
        self.reduce()


    def filter(self, wikipedia_loc):
        self.remove_non_informative_classes()
        # self.store_intermediate_graphs(f"../results/wikc_in_llms/{self.model_name}/intermediate_graphs", step="non_informative")
        self.remove_specific_top_level_classes()
        # self.store_intermediate_graphs(f"../results/wikc_in_llms/{self.model_name}/intermediate_graphs", step="specific_top_level")

        # may remove some important classes
        wikc = self.remove_rare_classes(wikipedia_loc, depth=3)
        return wikc


    def remove_non_informative_classes(self):
        ''' Removes non-informative classes: 
            classes with only one superclass, one subclass, and without direct instances
        '''
        non_info_cls = graph_utils.find_non_informative_cls(self.wiki_dag, self.cls_inst_stats)
        while len(non_info_cls) > 0:
            # remove non-informative classes recursively
            for node in tqdm(non_info_cls, desc='Filter non-informative classes'):
                parent = list(self.wiki_dag.predecessors(node))[0]
                child = list(self.wiki_dag.successors(node))[0]
                self.wiki_dag.remove_node(node)
                if not nx.has_path(self.wiki_dag, parent, child):
                    self.wiki_dag.add_edge(parent, child)
            non_info_cls = graph_utils.find_non_informative_cls(self.wiki_dag, self.cls_inst_stats)
        self.reduce()

    
    def remove_specific_top_level_classes(self):
        '''
        Removes top-level classes that are too specific: e.g., testbed -> entity
        '''
        topcls = list(self.wiki_dag.successors(self.root))
        nodes_to_remove = set()
        for cls in tqdm(topcls, desc='Filter specific top-level classes'):
            redundancy = True
            for child in self.wiki_dag.successors(cls):
                parents = set(self.wiki_dag.predecessors(child)) - set(topcls)
                if len(parents) == 0:
                    redundancy = False
                    break
            if len(list(self.wiki_dag.successors(cls))) == 0:
                print(f"        Warning: Top-level class {cls} has no children.")
            # remove topclasses that have no children
            if redundancy:
                self.wiki_dag.remove_node(cls) # remove top-level class
                nodes_to_remove.add(cls)
                self.cls_inst_stats.pop(cls, None)
        print(f"    Current number of top-level classes: {len(list(self.wiki_dag.successors(self.root)))}")
        print(f"    Removed {len(nodes_to_remove)} specific top-level classes: {nodes_to_remove}")
    
    
    def remove_rare_classes(self, wikipedia_loc, depth=3):
        ''' 
        Removes classes that (1) either without cumulative instances (include itself)
            or (2) without a Wikipedia page (multiple languages)

        Args:
            wikipedia_loc (str): Path to the Wikipedia mapped entities file.
            depth (int): Depth to keep for top-level classes.
        '''
        # (1) filter classes without cumulative instances
        cum_stats = graph_utils.cumulative_stats(self.cls_inst_stats, nx.to_dict_of_lists(self.wiki_dag.reverse()))
        nodes = list(self.wiki_dag.nodes())
        longtail_count = 0
        for cls in tqdm(nodes, desc='Filter classes without cumulative instances'):
            if cum_stats[cls] < 1:
                self.wiki_dag.remove_node(cls)
                self.cls_inst_stats.pop(cls, None)
                longtail_count += 1
        print(f"    Removed {longtail_count} classes without any cumulative instances.")
        # self.store_intermediate_graphs(f"../results/wikc_in_llms/{self.model_name}/intermediate_graphs", step="rare_classes_instances")

        # (2) filter classes without a Wikipedia page
        mapped_ents = graph_utils.load_wikipedia_mapped_ents(self.wiki_dag, wikipedia_loc)
        topcls = nx.single_source_shortest_path_length(self.wiki_dag, source=self.root, cutoff=depth).keys()
        mapped_ents.update(topcls)
        mapped_wiki_ents = mapped_ents & set(self.wiki_dag.nodes())
        # rebuild graph based on mapped wiki classes
        wikc = nx.DiGraph()
        for node in mapped_wiki_ents:
            wikc.add_node(node)
        for node in tqdm(mapped_wiki_ents, desc='Filter classes without a Wikipedia page'):
            ancestors = list(graph_utils.get_first_ancestors_for_rebuild(self.wiki_dag, node, mapped_wiki_ents))
            for ancestor in ancestors:
                wikc.add_edge(ancestor, node) # parent -> child
        wikc = nx.transitive_reduction(wikc)
        return wikc
    
    def save_mapping(self, loc):
        if not os.path.exists(loc):
            raise FileNotFoundError(f"Directory does not exist: {loc}")
        with open(os.path.join(loc, f'{self.model_name}_mapping.txt'), 'w') as f:
            for child, parent in self.mapping.items():
                f.write(f"{child}\t{parent}\n")

    def stats(self):
        """
        Returns the statistics of the taxonomy.
        """
        return {
            "number_of_nodes": self.wiki_dag.number_of_nodes(),
            "number_of_edges": self.wiki_dag.number_of_edges(),
            "max_depth": max(nx.shortest_path_length(self.wiki_dag, source='wd:Q35120').values()),
            "weakly_connected": nx.is_weakly_connected(self.wiki_dag),
            "directed_acyclic": nx.is_directed_acyclic_graph(self.wiki_dag),
            "number_of_roots": len([node for node in self.wiki_dag.nodes() if not list(self.wiki_dag.predecessors(node))]),
            "number_of_leaves": len([node for node in self.wiki_dag.nodes() if self.wiki_dag.out_degree(node) == 0]),
            "number_of_internal_nodes": len([node for node in self.wiki_dag.nodes() if self.wiki_dag.out_degree(node) > 0]),
            "average_in_degree": sum(dict(self.wiki_dag.in_degree()).values()) / self.wiki_dag.number_of_nodes(),
        }

    
    def store_intermediate_graphs(self, loc, step=None):
        if not os.path.exists(loc):
            os.makedirs(loc)
        if step is None:
            raise ValueError("Step is required to store intermediate graphs.")
        # store current taxonomy
        with open(os.path.join(loc, f'{self.model_name}_{step}.txt'), "w") as f:
            for u, v in self.wiki_dag.edges():
                f.write(f"{v}\t{u}\n") # child -> parent subclassOf