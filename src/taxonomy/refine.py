from clean import TaxonCleaner
from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm
import networkx as nx
import graph_utils
import pickle
import sys
import os
import numpy as np


def satisfy_constraints(graph, stage):
    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError(f"The graph is not a DAG after {stage}.")
    if not nx.is_weakly_connected(graph):
        raise ValueError(f"The graph is not weakly connected after {stage}.")
    num_roots = len([node for node in graph.nodes() if not list(graph.predecessors(node))])
    if num_roots != 1:
        raise ValueError(f"The graph has {num_roots} root nodes after {stage}.")
    # print statistics
    print(f"After {stage}:")
    print(f"  Number of nodes: {graph.number_of_nodes()}")
    print(f"  Number of edges: {graph.number_of_edges()}")
    print(f"  Max depth: {max(nx.shortest_path_length(graph, source='wd:Q35120').values())}")
    avg_in_degree = sum(dict(graph.in_degree()).values()) / graph.number_of_nodes()
    print(f"  Average in-degree: {avg_in_degree}")
    
    

def draw_intermediate_graphs(graph, cls2label, mapping, stage, loc):
    nodes_list = ['wd:Q7930989', 'wd:Q215627', 'wd:Q15324', 'wd:Q476300', 'wd:Q11424', 'wd:Q46970', 'wd:Q783794', 'wd:Q2095', 'wd:Q515', 'wd:Q5']
    nodes_name = [cls2label[node] for node in nodes_list]
    for i, node in enumerate(nodes_list):
        # check folder exists
        if not os.path.exists(os.path.join(loc, nodes_name[i])):
            os.makedirs(os.path.join(loc, nodes_name[i]))
        image = graph_utils.draw_graph(graph, node, cls2label, mapping=mapping)
        if image is None:
            continue
        with open(os.path.join(loc, nodes_name[i], f"{node[3:]}_{stage}.svg"), "w") as svg_file:
            svg_file.write(image)


def load_submatrix(emb_pkl_path, sub_nodes):
    """
    Load a pre-computed global embedding matrix and extract the sub-matrix
    for a subset of nodes (e.g. a specific model's taxonomy).

    The pkl file is expected to contain {"nodes": [...], "emb": np.ndarray}.

    Args:
        emb_pkl_path : path to the global embedding pickle file
        sub_nodes    : list of node ids required (e.g. sorted(dag.nodes()))

    Returns:
        emb_sub : (len(sub_nodes), d) numpy array, rows aligned with sub_nodes
    """
    with open(emb_pkl_path, 'rb') as f:
        data = pickle.load(f)
    global_nodes = data["nodes"]
    global_emb = data["emb"]

    if not global_nodes[0].startswith('wd:'):
        global_nodes = ['wd:' + node for node in global_nodes]

    global_node2idx = {v: i for i, v in enumerate(global_nodes)}

    missing = [v for v in sub_nodes if v not in global_node2idx]
    if missing:
        raise KeyError(
            f"{len(missing)} nodes not found in global embeddings, "
            f"first 5: {missing[:5]}"
        )

    indices = np.array([global_node2idx[v] for v in sub_nodes])
    return global_emb[indices]



if __name__ == "__main__":

    # get arguments
    if len(sys.argv) > 1:
        llm = sys.argv[1]  # e.g., 'gpt-4o'
    else:
        raise ValueError("Please provide the LLM name as a command-line argument.")
    
    cls2label = {} # qid: label
    path = '../../data/clean'
    with open(os.path.join(path, 'wiki_2026_filtered_labels_v3.tsv'), 'r') as f_label: # TBC
        for line in f_label:
            terms = line.strip().split('\t')
            if len(terms) > 1:
                cls, label = terms[0], terms[1]
                if not cls.startswith('wd:'):
                    cls = 'wd:' + cls
                cls2label[cls] = label[1:-1]
    
    # parameters
    threshold = 0.5

    cleaner = TaxonCleaner('../../quick_check/data/clean/', model=llm)
    # cleaner.deduce_predictions(f'../results/clean/{llm}_outputs_2026.json', threshold=threshold) #TBC
    # cleaner.store_predictions(f"../results/wikc_in_llms/{llm}")
    # cleaner.store_majority_predictions(f"../results/wikc_in_llms/", model_names=['qwen32b', 'gemma27b', 'mistral24b'])
    cleaner.load_majority_predictions(f"../results/wikc_in_llms/")


    # calculate similarity matrix
    emb_path = '../../data/clean'
    emb_pkl = os.path.join(emb_path, 'wiki_2026_labels_emb.pkl')
    all_nodes = sorted(list(cleaner.wiki_dag.nodes()))
    if os.path.exists(emb_pkl):
        emb_sub = load_submatrix(emb_pkl, all_nodes)
        simi_matrix = emb_sub @ emb_sub.T
        node2id = {node: i for i, node in enumerate(all_nodes)}
    else:
        model_name = 'Lihuchen/pearl_small' # or 'sentence-transformers/LaBSE'
        Str_model = AutoModel.from_pretrained(model_name)
        Str_tokenizer = AutoTokenizer.from_pretrained(model_name)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        Str_model.to(device) # gpu
        print("Calculating similarity matrix...")
        simi_matrix, node2id = graph_utils.calculate_simi_matrix(cleaner.wiki_dag, Str_model, Str_tokenizer, cls2label, path=f"../results/wikc_in_llms/{llm}")
    
    # original taxonomy
    satisfy_constraints(cleaner.wiki_dag, "original")
    # draw_intermediate_graphs(cleaner.wiki_dag, cls2label, cleaner.mapping, "original", f"../results/wikc_in_llms/{llm}")

    # cut
    cleaner.cut(simi_matrix, node2id)
    satisfy_constraints(cleaner.wiki_dag, "cut")
    draw_intermediate_graphs(cleaner.wiki_dag, cls2label, cleaner.mapping, "cut", f"../results/wikc_in_llms/{llm}")
    cleaner.store_intermediate_graphs(f"../results/wikc_in_llms/{llm}/intermediate_graphs", step="cut")


    # resolve
    cleaner.resolve(simi_matrix, node2id)
    satisfy_constraints(cleaner.wiki_dag, "resolve")
    draw_intermediate_graphs(cleaner.wiki_dag, cls2label, cleaner.mapping, "resolve", f"../results/wikc_in_llms/{llm}")
    cleaner.store_intermediate_graphs(f"../results/wikc_in_llms/{llm}/intermediate_graphs", step="resolve")

    # reduce
    cleaner.reduce()
    satisfy_constraints(cleaner.wiki_dag, "reduce")
    draw_intermediate_graphs(cleaner.wiki_dag, cls2label, cleaner.mapping, "reduce", f"../results/wikc_in_llms/{llm}")
    cleaner.store_intermediate_graphs(f"../results/wikc_in_llms/{llm}/intermediate_graphs", step="reduce")

    # merge_rewire
    rewire_loc = f"../results/rewire/results_json/{llm}_rewire_outputs_2026.json"
    save_rewire_loc = f"../results/rewire/data/"
    if os.path.exists(rewire_loc):
        unpredicted_edges = cleaner.exist_prediction_for_all_reiwred_links(rewire_loc, simi_matrix=simi_matrix, node2id=node2id)
        if len(unpredicted_edges) > 0:
            print(f"Number of unpredicted edges: {len(unpredicted_edges)}")
            # save the unpredicted edges
            with open(f"../results/wikc_in_llms/{llm}/unpredicted_edges.txt", "w") as f:
                for edge in unpredicted_edges:
                    child, parent = edge
                    f.write(f"{child}\t{parent}\n")
            raise ValueError("Unpredicted edges found, please check the rewire results.")
    valid_merge_edges = cleaner.check_valid_merge_edges(rewire_file_loc=rewire_loc, threshold_rewire=0.5, simi_matrix=simi_matrix, node2id=node2id, save_loc=save_rewire_loc)
    cleaner.merge_new_4(valid_merge_edges) # parent, child can only have one merge at maximum
    satisfy_constraints(cleaner.wiki_dag, "merge")
    draw_intermediate_graphs(cleaner.wiki_dag, cls2label, cleaner.mapping, "merge", f"../results/wikc_in_llms/{llm}")
    cleaner.store_intermediate_graphs(f"../results/wikc_in_llms/{llm}/intermediate_graphs", step="merge")

    # filter
    wikc = cleaner.filter(wikipedia_loc='../../data/data_2026/wikidata/wikipedia')
    satisfy_constraints(wikc, "filter")
    draw_intermediate_graphs(wikc, cls2label, cleaner.mapping, "filter", f"../results/wikc_in_llms/{llm}")
    cleaner.store_intermediate_graphs(f"../results/wikc_in_llms/{llm}/intermediate_graphs", step="filter")

    # store the final cleaned taxonomy
    with open(f"../results/wikc_in_llms/{llm}/wicleanData.txt", "w") as f:
        for u, v in wikc.edges():
            f.write(f"{v}\t{u}\n") # child -> parent subclassOf

    # show the final statistics
    print(f"================================================")
    print(f"Number of nodes: {wikc.number_of_nodes()}")
    print(f"Number of edges: {wikc.number_of_edges()}")
    print(f"Max depth: {max(nx.shortest_path_length(wikc, source='wd:Q35120').values())}")
    print(f"Number of roots: {len([node for node in wikc.nodes() if not list(wikc.predecessors(node))])}")
    print(f"Number of leaves: {len([node for node in wikc.nodes() if wikc.out_degree(node) == 0])}")
    print(f"Number of internal nodes: {len([node for node in wikc.nodes() if wikc.out_degree(node) > 0])}")
    
    avg_in_degree = sum(dict(wikc.in_degree()).values()) / wikc.number_of_nodes()
    avg_out_degree = sum(dict(wikc.out_degree()).values()) / wikc.number_of_nodes()
    print(f"Average in-degree: {avg_in_degree}")
    print(f"Average out-degree: {avg_out_degree}")
    print(f"================================================")

    # store deleted edges
    with open(f"../results/wikc_in_llms/{llm}/deleted_edges.txt", "w") as f:
        for parent, child in cleaner.edges_del:
            f.write(f"{parent}\t{child}\n")

    # store mapping: child -> parent
    cleaner.save_mapping(f"../results/wikc_in_llms/{llm}")

    # store new cls_inst_count
    with open(f"../results/wikc_in_llms/{llm}/cls_inst_count.txt", "w") as f:
        for cls, count in cleaner.cls_inst_stats.items():
            f.write(f"{cls}\t{count}\n")
    
    # print the top level classes
    top_level_classes = list(wikc.successors('wd:Q35120'))
    print(f"Number of top level classes: {len(top_level_classes)}")
    for cls in top_level_classes:
        print(f"{cls}: {cls2label[cls]}")
    
