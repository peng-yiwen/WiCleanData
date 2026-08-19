from clean import TaxonCleaner
from transformers import AutoTokenizer, AutoModel
import torch
import networkx as nx
import graph_utils
import config
import pickle
import csv
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
    print(f"  Max depth: {max(nx.shortest_path_length(graph, source='Q35120').values())}")
    avg_in_degree = sum(dict(graph.in_degree()).values()) / graph.number_of_nodes()
    print(f"  Average in-degree: {avg_in_degree}")
    
    

def draw_intermediate_graphs(graph, cls2label, mapping, stage, loc):
    nodes_list = ['Q7930989', 'Q215627', 'Q15324', 'Q476300', 'Q11424', 'Q46970', 'Q783794', 'Q2095', 'Q515', 'Q5']
    nodes_name = [cls2label[node] for node in nodes_list]
    for i, node in enumerate(nodes_list):
        # check folder exists
        if not os.path.exists(os.path.join(loc, nodes_name[i])):
            os.makedirs(os.path.join(loc, nodes_name[i]))
        image = graph_utils.draw_graph(graph, node, cls2label, mapping=mapping)
        if image is None:
            continue
        with open(os.path.join(loc, nodes_name[i], f"{node}_{stage}.svg"), "w") as svg_file:
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

    # No wd: prefix for global nodes
    # if not global_nodes[0].startswith('wd:'):
    #     global_nodes = ['wd:' + node for node in global_nodes]

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

    # file paths
    OUTPUT_FOLDER = config.LLM_OUTPUT_DIR
    INTERMEDIATE_FOLDER = config.INTERMEDIATE_GRAPHS_DIR
    LABELS_FILE = config.TAXONOMY_LABELS_FILE
    CLS_INST_COUNT_FILE = config.CLS_INST_COUNT_FILE
    HIERARCHY_FILE = config.TAXONOMY_FILE
    MAJORITY_PREDICTIONS_FILE = config.MAJORITY_PREDICTIONS_FILE
    MAJORITY_PREDICTIONS_REWIRE_FILE = config.MAJORITY_PREDICTIONS_REWIRE_FILE
    cls2label = graph_utils.load_labels(LABELS_FILE)
    

    CLS_INST_COUNT = dict()
    with open(CLS_INST_COUNT_FILE, "r", encoding="utf-8") as f:
        reader = csv.reader(f) # "class", "instance_count"
        for row in reader:
            if row[0].startswith("http://www.wikidata.org/entity/"):
                cls = row[0].split("/")[-1]
                CLS_INST_COUNT[cls] = int(row[1])
    
    # parameters
    LLMs = config.LLM_MODELS
    threshold = 0.5

    # get majority predictions
    cleaner = TaxonCleaner(output_dir=OUTPUT_FOLDER, init_taxonomy=HIERARCHY_FILE, 
                                 cls_inst_count=CLS_INST_COUNT, models=LLMs)
    if not os.path.exists(os.path.join(OUTPUT_FOLDER, MAJORITY_PREDICTIONS_FILE)):
        cleaner.get_majority_predictions(file_name=MAJORITY_PREDICTIONS_FILE, threshold=threshold)
    else:
        cleaner.load_majority_predictions(file_name=MAJORITY_PREDICTIONS_FILE)


    # calculate similarity matrix
    emb_pkl = config.EMBEDDING_PKL_FILE
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
        simi_matrix, node2id = graph_utils.calculate_simi_matrix(cleaner.wiki_dag, Str_model, Str_tokenizer, cls2label, file_path=emb_pkl)
    
    # original taxonomy
    satisfy_constraints(cleaner.wiki_dag, "original")
    draw_intermediate_graphs(cleaner.wiki_dag, cls2label, cleaner.mapping, "original", INTERMEDIATE_FOLDER)

    # cut
    cleaner.cut(simi_matrix, node2id)
    satisfy_constraints(cleaner.wiki_dag, "cut")
    draw_intermediate_graphs(cleaner.wiki_dag, cls2label, cleaner.mapping, "cut", INTERMEDIATE_FOLDER)
    cleaner.store_intermediate_graphs(INTERMEDIATE_FOLDER, step="cut")


    # resolve
    cleaner.resolve(simi_matrix, node2id)
    satisfy_constraints(cleaner.wiki_dag, "resolve")
    draw_intermediate_graphs(cleaner.wiki_dag, cls2label, cleaner.mapping, "resolve", INTERMEDIATE_FOLDER)
    cleaner.store_intermediate_graphs(INTERMEDIATE_FOLDER, step="resolve")

    # reduce
    cleaner.reduce()
    satisfy_constraints(cleaner.wiki_dag, "reduce")
    draw_intermediate_graphs(cleaner.wiki_dag, cls2label, cleaner.mapping, "reduce", INTERMEDIATE_FOLDER)
    cleaner.store_intermediate_graphs(INTERMEDIATE_FOLDER, step="reduce")

    # merge_rewire
    rewire_loc = os.path.join(OUTPUT_FOLDER, MAJORITY_PREDICTIONS_REWIRE_FILE)
    if os.path.exists(rewire_loc):
        unpredicted_edges = cleaner.exist_prediction_for_all_reiwred_links(rewire_loc, simi_matrix=simi_matrix, node2id=node2id)
        if len(unpredicted_edges) > 0:
            # save the unpredicted edges
            with open(os.path.join(OUTPUT_FOLDER, "unpredicted_rewire_edges.txt"), "w") as f:
                for edge in unpredicted_edges:
                    child, parent = edge
                    f.write(f"{child},{parent}\n")
            raise ValueError("Unpredicted edges found, please check the rewire results.")
    valid_merge_edges = cleaner.check_valid_merge_edges(rewire_file_loc=rewire_loc, threshold_rewire=threshold, simi_matrix=simi_matrix, node2id=node2id, save_loc=rewire_loc)
    cleaner.merge(valid_merge_edges) # parent, child can only have one merge at maximum
    satisfy_constraints(cleaner.wiki_dag, "merge")
    draw_intermediate_graphs(cleaner.wiki_dag, cls2label, cleaner.mapping, "merge", INTERMEDIATE_FOLDER)
    cleaner.store_intermediate_graphs(INTERMEDIATE_FOLDER, step="merge")

    # filter
    wikc = cleaner.filter(wikipedia_loc=config.WIKIPEDIA_DIR)
    satisfy_constraints(wikc, "filter")
    draw_intermediate_graphs(wikc, cls2label, cleaner.mapping, "filter", INTERMEDIATE_FOLDER)
    cleaner.store_intermediate_graphs(INTERMEDIATE_FOLDER, step="filter")

    # store the taxonomy before wikipedia filtering
    with open(config.WICLEAN_TAXONOMY_BEFORE_WP_FILE, "w") as f:
        for u, v in wikc.edges():
            f.write(f"{v},{u}\n") # child -> parent subclassOf

    # store the final cleaned taxonomy
    os.makedirs(config.WICLEAN_OUTPUT_DIR, exist_ok=True)
    with open(config.WICLEAN_TAXONOMY_FILE, "w") as f:
        for u, v in wikc.edges():
            f.write(f"{v},{u}\n") # child -> parent subclassOf

    # show the final statistics
    print(f"================================================")
    print(f"Number of nodes: {wikc.number_of_nodes()}")
    print(f"Number of edges: {wikc.number_of_edges()}")
    print(f"Max depth: {max(nx.shortest_path_length(wikc, source='Q35120').values())}")
    print(f"Number of roots: {len([node for node in wikc.nodes() if not list(wikc.predecessors(node))])}")
    print(f"Number of leaves: {len([node for node in wikc.nodes() if wikc.out_degree(node) == 0])}")
    print(f"Number of internal nodes: {len([node for node in wikc.nodes() if wikc.out_degree(node) > 0])}")
    
    avg_in_degree = sum(dict(wikc.in_degree()).values()) / wikc.number_of_nodes()
    avg_out_degree = sum(dict(wikc.out_degree()).values()) / wikc.number_of_nodes()
    print(f"Average in-degree: {avg_in_degree}")
    print(f"Average out-degree: {avg_out_degree}")
    print(f"================================================")

    # # store deleted edges
    # with open(os.path.join(DATA_FOLDER, "wicleanData/", "deleted_edges.txt"), "w") as f:
    #     for parent, child in cleaner.edges_del:
    #         f.write(f"{parent}\t{child}\n")

    # store mapping: child -> parent
    cleaner.save_mapping(config.WICLEAN_MAPPING_FILE)

    # store new cls_inst_count
    with open(os.path.join(config.WICLEAN_OUTPUT_DIR, "cls_inst_count.csv"), "w") as f:
        for cls, count in cleaner.cls_inst_stats.items():
            f.write(f"{cls}\t{count}\n") # no wd: prefix
    
    # print the top level classes
    top_level_classes = list(wikc.successors('Q35120'))
    print(f"Number of top level classes: {len(top_level_classes)}")
    for cls in top_level_classes:
        print(f"{cls}: {cls2label[cls]}")
    
