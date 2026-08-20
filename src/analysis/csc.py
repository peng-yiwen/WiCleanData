import numpy as np
import networkx as nx
import pickle
import os
from itertools import combinations
from collections import defaultdict
from scipy.special import basic
from scipy.stats import kendalltau
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel

os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ──────────────────────────────────────────────
#  Wu & Palmer pairwise score (original helper)
# ──────────────────────────────────────────────

def wu_p_score(dag, root):
    """
    Compute the Wu & Palmer similarity matrix for all concept pairs
    using their label embeddings.

    Returns:
        wp_matrix – (N, N) numpy array of Wu & Palmer similarities
    """
    nodes = sorted(dag.nodes())
    n = len(nodes)
    # node2idx = {v: i for i, v in enumerate(nodes)}
    depth = nx.shortest_path_length(dag, source=root)

    path2root = dict()
    for v in nodes:
        path2root[v] = nx.ancestors(dag, v)
        
    wp_matrix = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            ca, cb = nodes[i], nodes[j]
            common = path2root[ca] & path2root[cb]
            if not common:
                continue
            lca_depth = max(depth[node]+1 for node in common)
            # lca_depth = max(lca_depth, 1) # the root depth is 1
            score = 2.0 * lca_depth / (depth[ca] + depth[cb] + 2)
            wp_matrix[i, j] = score
            wp_matrix[j, i] = score

    # fill the diagonal with 1
    np.fill_diagonal(wp_matrix, 1.0)

    return nodes, wp_matrix

# ──────────────────────────────────────────────
#  Semantic similarity matrix
# ──────────────────────────────────────────────

def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def encode_texts(model, tokenizer, texts, device, batch_size=128):
    """Encode a list of strings into L2-normalised embeddings."""
    all_embs = []
    for i in tqdm(range(0, len(texts), batch_size), desc="    Encoding texts"):
        batch = texts[i:i + batch_size]
        batch_dict = tokenizer(
            batch, max_length=512, padding=True, truncation=True, return_tensors='pt'
        ).to(device)
        with torch.no_grad():
            outputs = model(**batch_dict)
        emb = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
        emb = F.normalize(emb, p=2, dim=1)
        all_embs.append(emb.cpu())
    return torch.cat(all_embs, dim=0).numpy()


def compute_semantic_similarity(nodes, cls2label, model_name='Lihuchen/pearl_small'):
    """
    Compute cosine similarity matrix for all concept pairs
    using their label embeddings.

    Returns:
        simi_matrix – (N, N) numpy array of cosine similarities
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    labels = [cls2label.get(node, node) for node in nodes]
    print(f"  Encoding {len(labels)} concept labels with {model_name} ...")
    emb_matrix = encode_texts(model, tokenizer, labels, device)
    # cosine similarity (embeddings are already L2-normalised)
    simi_matrix = emb_matrix @ emb_matrix.T
    return simi_matrix


# ──────────────────────────────────────────────
#  CSC metric
# ──────────────────────────────────────────────

def compute_csc(semantic_matrix, taxonomic_matrix):
    """
    Concept Similarity Correlation:
        CSC = kendall_tau(S, W)
    computed over all unique pairs (upper triangle of the matrices).
    """
    n = semantic_matrix.shape[0]
    iu = np.triu_indices(n, k=1)
    sem_vals = semantic_matrix[iu]
    tax_vals = taxonomic_matrix[iu]
    tau, p_value = kendalltau(sem_vals, tax_vals)
    return tau, p_value



# ──────────────────────────────────────────────
#  Sampled CSC (memory-efficient for large graphs)
# ──────────────────────────────────────────────

def precompute_ancestors(dag, nodes):
    """Pre-compute the ancestor set for every node in the DAG."""
    ancestors = {}
    for v in tqdm(nodes, desc="  Pre-computing ancestors"):
        ancestors[v] = nx.ancestors(dag, v)
    return ancestors


def wu_palmer_single_pair(ca, cb, ancestors, depth):
    """
    Wu & Palmer similarity for a single pair (ca, cb).
    depth[v] = shortest-path distance from root (root=0).
    """
    common = ancestors[ca] & ancestors[cb]
    if not common:
        return 0.0
    lca_depth = max(depth[node] + 1 for node in common)
    return 2.0 * lca_depth / (depth[ca] + depth[cb] + 2)


def compute_embeddings(nodes, cls2label, model_name='Lihuchen/pearl_small'):
    """
    Compute L2-normalised embedding matrix (N, d) for all nodes.
    Unlike compute_semantic_similarity, this does NOT build the N×N
    cosine matrix, so memory stays at O(N*d).

    Returns:
        emb_matrix – (N, d) numpy float32 array
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    labels = [cls2label.get(node, node) for node in nodes]
    print(f"  Encoding {len(labels)} concept labels with {model_name} ...")
    emb_matrix = encode_texts(model, tokenizer, labels, device)
    return emb_matrix


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

    global_node2idx = {v: i for i, v in enumerate(global_nodes)}

    missing = [v for v in sub_nodes if v not in global_node2idx]
    if missing:
        raise KeyError(
            f"{len(missing)} nodes not found in global embeddings, "
            f"first 5: {missing[:5]}"
        )

    indices = np.array([global_node2idx[v] for v in sub_nodes])
    return global_emb[indices]


def _compute_tau_for_nodes(node_indices, nodes, emb_matrix, ancestors, depth):
    """
    Compute Kendall τ over all pairs of a sampled node subset.

    Semantic similarity is a single (k, k) matrix multiply.
    Wu-Palmer is computed for all k*(k-1)/2 pairs.
    """
    sub_nodes = [nodes[i] for i in node_indices]
    sub_emb = emb_matrix[node_indices]          # (k, d)

    # Semantic: one matrix multiply → full (k, k) cosine matrix
    sem_matrix = sub_emb @ sub_emb.T            # (k, k)

    # Wu-Palmer: (k, k) matrix
    k = len(node_indices)
    wp_matrix = np.zeros((k, k), dtype=np.float32)
    for i in range(k):
        for j in range(i + 1, k):
            wp_matrix[i, j] = wu_palmer_single_pair(
                sub_nodes[i], sub_nodes[j], ancestors, depth
            )
            wp_matrix[j, i] = wp_matrix[i, j]
    np.fill_diagonal(wp_matrix, 1.0)

    # Extract upper triangle (all unique pairs) and compute Kendall τ
    iu = np.triu_indices(k, k=1)
    tau, p_value = kendalltau(sem_matrix[iu], wp_matrix[iu])
    return tau, p_value


def compute_csc_sampled(dag, root, cls2label, k=20000, m=50,
                        model_name='Lihuchen/pearl_small', seed=42, emb_path=None):
    """
    Memory-efficient CSC via repeated random node sampling.

    Each round randomly selects k nodes, then computes all k*(k-1)/2
    pair similarities via matrix operations. The final CSC is the
    average Kendall τ across m rounds.

    Memory per round: two (k, k) float32 matrices = 2 * k^2 * 4 bytes.
    e.g. k=5000 → ~200 MB per round;  k=10000 → ~800 MB per round.

    Args:
        dag       : nx.DiGraph – taxonomy (edges: parent → child)
        root      : root node id
        cls2label : dict, node → label string
        k         : number of nodes to sample per round
        m         : number of sampling rounds
        model_name: HuggingFace model for embeddings
        seed      : base random seed (each round uses seed+round_idx)
        emb_path  : optional path to pre-computed embeddings (pickle)

    Returns:
        tau_mean : float  – mean Kendall τ across m rounds
        tau_std  : float  – standard deviation of τ across m rounds
        tau_list : list   – individual τ values for each round
    """
    nodes = sorted(dag.nodes())
    n = len(nodes)
    k_actual = min(k, n)
    n_pairs = k_actual * (k_actual - 1) // 2

    if k_actual >= n:
        m = 1
        print(f"  k={k} ≥ N={n}  →  using all nodes, exact computation (1 round)")
    else:
        print(f"  N={n}, sampling {k_actual} nodes × {m} rounds "
              f"({n_pairs} pairs per round)")

    # ── One-time pre-computation ──
    depth = nx.shortest_path_length(dag, source=root)
    ancestors = precompute_ancestors(dag, nodes)
    if os.path.exists(os.path.join(emb_path, 'wiki_labels_emb.pkl')):
        with open(os.path.join(emb_path, 'wiki_emb.pkl'), 'rb') as f:
            data = pickle.load(f)
            assert len(data["nodes"]) == n, "Node number mismatch!"
            # assert data["nodes"] == nodes, "Node order mismatch!"
            nodes = data["nodes"]
            emb_matrix = data["emb"]
    else:
        emb_matrix = compute_embeddings(nodes, cls2label, model_name)
        with open(os.path.join(emb_path, 'wiki_labels_emb.pkl'), 'wb') as f:
            pickle.dump({"nodes": nodes, "emb": emb_matrix}, f)

    # ── Repeated sampling rounds ──
    tau_list = []
    for r in range(m):
        rng = np.random.default_rng(seed + r)

        if k_actual >= n:
            node_indices = np.arange(n)
        else:
            node_indices = rng.choice(n, size=k_actual, replace=False)

        tau, p_value = _compute_tau_for_nodes(
            node_indices, nodes, emb_matrix, ancestors, depth
        )
        tau_list.append(tau)
        print(f"    Round {r+1}/{m}: τ = {tau:.4f}  (p = {p_value:.2e})")

    tau_mean = float(np.mean(tau_list))
    tau_std = float(np.std(tau_list))
    print(f"  ── CSC (mean ± std over {m} rounds): {tau_mean:.4f} ± {tau_std:.4f}")

    return tau_mean, tau_std, tau_list


# ──────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────

if __name__ == "__main__":
    root = 'Q35120'

    DATA_PATH = '../../data/wikidata/'

    cls2label = {} # qid: label
    with open(os.path.join(DATA_PATH, 'wiki_taxonomy_extracted_labels.tsv'), 'r') as f_label:
        for line in f_label:
            # wd:Q96196524 rdfs:label "current entity" .
            terms = line.strip().split('\t')
            if len(terms) > 1:
                cls2label[terms[0]] = terms[1][1:-1]

    # # Wikidata: Use sampling
    # wiki_path = '../../data/wikidata/'
    # dag = nx.DiGraph()
    # with open(os.path.join(wiki_path,'wiki_taxonomy.tsv'), 'r') as taxoreader:
    #     for line in taxoreader:
    #         terms = line.strip().split('\t')
    #         if len(terms) > 3:
    #             child, parent = terms[0], terms[2]
    #             dag.add_edge(parent, child)
    # # emb_path = '../../results/wikc_in_llms/wikidata/Version-2026'
    # tau_mean, tau_std, tau_list = compute_csc_sampled(dag, root, cls2label, k=20000, m=50,
    #                                                   model_name='Lihuchen/pearl_small', seed=42, emb_path=wiki_path)

    models = [
        'gemma4b', 'gemma12b', 'gemma27b',
        'qwen8b', 'qwen14b', 'qwen32b',
        'mistral7b', 'mistral24b',
        'mixtral8x7b', 'llama8b',
    ]
    
    basic_path = '../../results/wikc_in_llms/'
    for model_name in tqdm(models, desc="Computing CSC for each model"):
        graph_path = os.path.join(basic_path, model_name, 'wikc.txt')
        if not os.path.exists(graph_path):
            raise ValueError(f"Graph {graph_path} not found.")
        
        dag = nx.DiGraph()
        with open(graph_path, 'r') as f:
            for line in f:
                child, parent = line.strip().split('\t')
                dag.add_edge(parent, child)

        print(f"\n{'='*50}")
        print(f" Model: {model_name}")
        print(f"{'='*50}")
        print(f"  Graph loaded: {dag.number_of_nodes()} nodes, {dag.number_of_edges()} edges")

        # Step 3: Wu & Palmer taxonomic similarity
        print("  Computing Wu & Palmer similarity matrix ...")
        nodes, wp_matrix = wu_p_score(dag, root)
        print(f"  Wu & Palmer matrix shape: {wp_matrix.shape}")

        # Step 1 & 2: Semantic similarity
        emb_path = '../../data/wikidata/'
        if os.path.exists(os.path.join(emb_path, 'wiki_labels_emb.pkl')):
            sem_matrix = load_submatrix(os.path.join(emb_path, 'wiki_labels_emb.pkl'), nodes)
            sem_simi_matrix = sem_matrix @ sem_matrix.T
        else:
            sem_simi_matrix = compute_semantic_similarity(nodes, cls2label)
        print(f"  Semantic similarity matrix shape: {sem_simi_matrix.shape}")

        # Step 4: CSC
        tau, p_value = compute_csc(sem_simi_matrix, wp_matrix)
        print(f"  CSC (Kendall τ) = {tau:.4f}  (p-value = {p_value:.2e})")
