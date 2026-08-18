from collections import defaultdict
from tqdm import tqdm
import networkx as nx
import graph_utils
import json
import os

class TaxonCleaner:
    def __init__(self, output_dir, init_taxonomy, cls_inst_count, models=['mistral24b']):
        # self.graph = graph
        self.root = "Q35120" # entity
        self.models_name = models
        # self.cls2label = None
        self.cls_inst_stats = cls_inst_count # direct instance count for each class
        self.irrel_pred = {} # list of tuples (parent, child, confidence)
        self.equiv_pred = {}
        self.reverse_pred = {}
        self.subsume_pred = {}
        self.rewire_links = None # dict of tuples (child, parent): list of hops for that tuple
        # self.data_path = data_path
        self.mapping = dict() # mapping from Wikidata to WiKC
        self.edges_del = set() # edges deleted during cleaning: parent -> child
        self.output_dir = output_dir
        self.wiki_dag = None # directed acyclic graph of the taxonomy
        self._init_taxonomy(init_taxonomy) # init the taxonomy by path file


    def _init_taxonomy(self, init_taxonomy):
        """ 
        Initializes the taxonomy from the data path.
        """
        taxonomy_top_to_down = defaultdict(set)
        with open(init_taxonomy, 'r') as topreader:
            for line in topreader:
                terms = line.strip().split(',')
                if len(terms) > 1:
                    child, parent = terms[0], terms[1] # no wd: prefix
                    taxonomy_top_to_down[parent].add(child)
        self.wiki_dag = nx.DiGraph(taxonomy_top_to_down)


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
        irrel_pred, equiv_pred, reverse_pred, subsume_pred = {}, {}, {}, {}
        with open(res_file, 'r') as f:
            llm_res = json.load(f)
        # Process llm_res to deduce predictions
        for res in llm_res:
            parent, child = res['id'].split('_')
            if res['plabel'].lower() == res['clabel'].lower():
                # if parent and child share the same label, consider it as equivalent -> exact match
                equiv_pred[tuple([parent, child])] = 1.0
                continue
            # deduce relationships based on LLM's answers
            if res['answer'].lower() == 'true' and res['answer_inverse'].lower() == 'false':
                subsume_pred[tuple([parent, child])] = min(res['confidence'], res['confidence_inverse'])
            elif res['answer'].lower() == 'true' and res['answer_inverse'].lower() == 'true':
                conf = min(res['confidence'], res['confidence_inverse'])
                if conf >= threshold:
                    equiv_pred[tuple([parent, child])] = conf
                else:
                    subsume_pred[tuple([parent, child])] = conf
            elif res['answer'].lower() == 'false' and res['answer_inverse'].lower() == 'true':
                conf = min(res['confidence'], res['confidence_inverse'])
                if conf >= threshold:
                    reverse_pred[tuple([parent, child])] = conf
                else:
                    subsume_pred[tuple([parent, child])] = conf
            elif res['answer'].lower() == 'false' and res['answer_inverse'].lower() == 'false':
                conf = min(res['confidence'], res['confidence_inverse'])
                if conf >= threshold:
                    irrel_pred[tuple([parent, child])] = conf
                else:
                    subsume_pred[tuple([parent, child])] = conf
            else: # LLM failed to answer, keep as it is
                subsume_pred[tuple([parent, child])] = min(res['confidence'], res['confidence_inverse'])
                print(f"LLM failed to answer for ({parent},{child}), forward:{res['answer']} / backward:{res['answer_inverse']}")

        # show statistics
        # total = len(irrel_pred) + len(equiv_pred) + len(reverse_pred) + len(subsume_pred)
        # print(f"Total predictions: {total}")
        # print(f"Irrelevant: {len(irrel_pred)}, Percentage: {len(irrel_pred) / total * 100:.2f}%")
        # print(f"Equivalent: {len(equiv_pred)}, Percentage: {len(equiv_pred) / total * 100:.2f}%")
        # print(f"Reverse: {len(reverse_pred)}, Percentage: {len(reverse_pred) / total * 100:.2f}%")
        # print(f"Subsuming: {len(subsume_pred)}, Percentage: {len(subsume_pred) / total * 100:.2f}%")

        # check if all edges in wiki_dag are in the predictions
        for edge in self.wiki_dag.edges():
            if edge not in irrel_pred and edge not in equiv_pred and edge not in reverse_pred and edge not in subsume_pred:
                raise ValueError(f"Edge {edge} not in any predictions.")
        return irrel_pred, equiv_pred, reverse_pred, subsume_pred

    
    def get_majority_predictions(self, file_name, threshold=0.5):
        # check if predictions of all models exist
        for model_name in self.models_name:
            if not os.path.exists(os.path.join(self.output_dir, f'{model_name}_predictions.json')):
                raise FileNotFoundError(f"Predictions file not found: {os.path.join(self.output_dir, f'{model_name}_predictions.json')}")
        # load all model predictions, then do majority voting
        model_predictions = dict()
        for model_name in self.models_name:
            model_predictions[model_name] = dict()
            irrel_pred, equiv_pred, reverse_pred, subsume_pred = self.deduce_predictions(os.path.join(self.output_dir, f'{model_name}_predictions.json'), threshold=threshold)
            for edge_, conf_ in irrel_pred.items():
                model_predictions[model_name][edge_] = ('[IRRELEVANT]', float(conf_))
            for edge_, conf_ in equiv_pred.items():
                model_predictions[model_name][edge_] = ('[EQUIVALENT]', float(conf_))
            for edge_, conf_ in reverse_pred.items():
                model_predictions[model_name][edge_] = ('[REVERSE]', float(conf_))
            for edge_, conf_ in subsume_pred.items():
                model_predictions[model_name][edge_] = ('[SUBSUME]', float(conf_))
        # do majority voting
        with open(os.path.join(self.output_dir, file_name), 'w') as f:
            for edge in self.wiki_dag.edges(): # parent -> child format edge
                pred_counts = dict()
                pred_confs = dict()
                for model_name in self.models_name:
                    if edge not in model_predictions[model_name]:
                        raise ValueError(f"Edge {edge} not found in {model_name} predictions.")
                    pred_, conf_ = model_predictions[model_name][edge]
                    pred_counts[pred_] = pred_counts.get(pred_, 0) + 1
                    pred_confs[pred_] = min(pred_confs.get(pred_, 1), conf_)
                # check if max_count is unique
                max_count = max(pred_counts.values())
                valid_pred = True
                if list(pred_counts.values()).count(max_count) > 1:
                    valid_pred = False
                if valid_pred:
                    max_count = max(pred_counts.values())
                    pred_final = [pred for pred, count in pred_counts.items() if count == max_count][0]
                    f.write(f"{edge[1]}_{edge[0]}\t{pred_final}\t{pred_confs[pred_final]}\n") # child -> parent format edge
                else:
                    f.write(f"{edge[1]}_{edge[0]}\t[SUBSUME]\t{1.0}\n")
        # load majority predictions
        self.irrel_pred, self.equiv_pred, self.reverse_pred, self.subsume_pred = {}, {}, {}, {}
        with open(os.path.join(self.output_dir, file_name), 'r') as f:
            for line in f:
                item_id, pred_final, conf = line.strip().split('\t')
                child, parent = item_id.split('_')
                if pred_final == '[IRRELEVANT]':
                    self.irrel_pred[tuple([parent, child])] = float(conf)
                elif pred_final == '[EQUIVALENT]':
                    self.equiv_pred[tuple([parent, child])] = float(conf)
                elif pred_final == '[REVERSE]':
                    self.reverse_pred[tuple([parent, child])] = float(conf)
                elif pred_final == '[SUBSUME]':
                    self.subsume_pred[tuple([parent, child])] = float(conf)
                else:
                    raise ValueError(f"Invalid prediction for {item_id}: {pred_final}, {conf}")
        # statistics
        total = len(self.irrel_pred) + len(self.equiv_pred) + len(self.reverse_pred) + len(self.subsume_pred)
        print(f"Total predictions: {total}")
        print(f"Irrelevant: {len(self.irrel_pred)}, Percentage: {len(self.irrel_pred) / total * 100:.2f}%")
        print(f"Equivalent: {len(self.equiv_pred)}, Percentage: {len(self.equiv_pred) / total * 100:.2f}%")
        print(f"Reverse: {len(self.reverse_pred)}, Percentage: {len(self.reverse_pred) / total * 100:.2f}%")
        print(f"Subsuming: {len(self.subsume_pred)}, Percentage: {len(self.subsume_pred) / total * 100:.2f}%")
        # check if all edges in wiki_dag are in the predictions
        for edge in self.wiki_dag.edges():
            if edge not in self.irrel_pred and edge not in self.equiv_pred and edge not in self.reverse_pred and edge not in self.subsume_pred:
                raise ValueError(f"Edge {edge} not in any predictions.")


    def load_majority_predictions(self, file_name):
        self.irrel_pred, self.equiv_pred, self.reverse_pred, self.subsume_pred = {}, {}, {}, {}
        with open(os.path.join(self.output_dir, file_name), 'r') as f:
            for line in f:
                item_id, pred_final, conf = line.strip().split('\t')
                child, parent = item_id.split('_')
                if pred_final == '[IRRELEVANT]':
                    self.irrel_pred[tuple([parent, child])] = float(conf)
                elif pred_final == '[EQUIVALENT]':
                    self.equiv_pred[tuple([parent, child])] = float(conf)
                elif pred_final == '[REVERSE]':
                    self.reverse_pred[tuple([parent, child])] = float(conf)
                elif pred_final == '[SUBSUME]':
                    self.subsume_pred[tuple([parent, child])] = float(conf)
                else:
                    raise ValueError(f"Invalid prediction for {item_id}: {pred_final}, {conf}")
        # statistics
        total = len(self.irrel_pred) + len(self.equiv_pred) + len(self.reverse_pred) + len(self.subsume_pred)
        print(f"Total predictions: {total}")
        print(f"Irrelevant: {len(self.irrel_pred)}, Percentage: {len(self.irrel_pred) / total * 100:.2f}%")
        print(f"Equivalent: {len(self.equiv_pred)}, Percentage: {len(self.equiv_pred) / total * 100:.2f}%")
        print(f"Reverse: {len(self.reverse_pred)}, Percentage: {len(self.reverse_pred) / total * 100:.2f}%")
        print(f"Subsuming: {len(self.subsume_pred)}, Percentage: {len(self.subsume_pred) / total * 100:.2f}%")
        # check if all edges in wiki_dag are in the predictions
        for edge in self.wiki_dag.edges():
            if edge not in self.irrel_pred and edge not in self.equiv_pred and edge not in self.reverse_pred and edge not in self.subsume_pred:
                raise ValueError(f"Edge {edge} not in any predictions.")
    
    ############################################################################################################################
    # Graph Transformation STRATEGY
    ############################################################################################################################

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
    
    
    ############################################################################################################################
    # MERGE & REWIRE STRATEGY
    ############################################################################################################################

    def get_reiwre_links(self, bfs_edges):
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
    

    def store_rewire_links(self, file_loc):
        # if not os.path.exists(file_loc):
        #     raise FileNotFoundError(f"File does not exist: {file_loc}")
        # store the rewire links
        total_rewire_edges = set()
        for k, v in self.rewire_links.items():
            total_rewire_edges.add(k) # include edge itself
            for edge in v: # hops
                total_rewire_edges.add(edge)
        with open(file_loc, 'w') as f:
            for edge in total_rewire_edges:
                child, parent = edge
                f.write(f"{child},{parent}\n") # split by ','
    
    
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
        self.get_reiwre_links(bfs_edges)
        # check if all edges are in the llm_rewire_res
        unpredicted_edges = set()
        for edge, hop in self.rewire_links.items():
            if edge not in llm_rewire_res:
                unpredicted_edges.add(edge) # child -> parent format edge
                continue
            for hop_edge in hop:
                if hop_edge not in llm_rewire_res:
                    unpredicted_edges.add(hop_edge) # child -> parent format edge
        return unpredicted_edges


    def check_valid_merge_edges(self, rewire_file_loc, threshold_rewire=0.5, save_loc=None, simi_matrix=None, node2id=None):
        bfs_edges = graph_utils.bfs_edges_by_level(self.wiki_dag, self.root)
        # sorted by similarity in descending order
        bfs_edges = graph_utils.reorder_edges_by_similarity(self.wiki_dag, bfs_edges, simi_matrix, node2id, reverse=True)
        if not os.path.exists(rewire_file_loc):
            self.get_reiwre_links(bfs_edges)
            if save_loc is None:
                raise ValueError("save_loc has to be provided if rewire_file_loc is not found.")
            self.store_rewire_links(save_loc) # save_loc has to be provided
            print(f"    Rewired links saved to {save_loc}")
            raise FileNotFoundError(f"Prediction for rewire links not found in {rewire_file_loc}")
        else:
            with open(rewire_file_loc, 'r') as f:
                rewire_res = json.load(f)
            llm_rewire_res = dict()
            for res in rewire_res:
                parent_, child_ = res['id'].split('_')
                llm_rewire_res[(child_, parent_)] = res # child -> parent format edge
        
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
            
            # store merge edges if it is valid
            if valid:
                valid_merge_edges.append((parent, child))
        print(f"    Found {len(valid_merge_edges)} valid merge edges.")
        print(f"    {cnt} nodes have more than one parent.")
        return valid_merge_edges

    
    def merge(self, valid_merge_edges: list):
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


    def filter(self, wikipedia_loc, depth=3):
        self.remove_non_informative_classes()
        self.remove_specific_top_level_classes()
        wikc = self.remove_rare_classes(wikipedia_loc, depth=depth) # may remove some important classes
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
    
    ############################################################################################################################
    # SAVE & STATISTICS
    ############################################################################################################################
    def save_mapping(self, file_loc):
        # if not os.path.exists(file_loc):
        #     raise FileNotFoundError(f"Directory does not exist: {file_loc}")
        with open(file_loc, 'w') as f:
            for child, parent in self.mapping.items():
                f.write(f"{child},{parent}\n")

    def stats(self):
        """
        Returns the statistics of the taxonomy.
        """
        return {
            "number_of_nodes": self.wiki_dag.number_of_nodes(),
            "number_of_edges": self.wiki_dag.number_of_edges(),
            "max_depth": max(nx.shortest_path_length(self.wiki_dag, source='Q35120').values()),
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
        with open(os.path.join(loc, f'{step}.txt'), "w") as f:
            for u, v in self.wiki_dag.edges():
                f.write(f"{v},{u}\n") # child -> parent subclassOf