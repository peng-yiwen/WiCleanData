import numpy as np
import networkx as nx
import pickle
import os
from itertools import combinations
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
import config

os.environ["TOKENIZERS_PARALLELISM"] = "false"

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


# ──────────────────────────────────────────────
#  Robustness metric
# ──────────────────────────────────────────────

def extract_groups(dag):
    """
    Identify groups at the lowest level of the taxonomy.

    A group is any non-leaf node that has at least one leaf child.
    Its characteristics are its direct leaf children.

    Returns:
        groups    : dict  {group_node: [leaf_child, ...]}
        all_chars : sorted list of all unique leaf nodes
    """
    leaves = {n for n in dag.nodes() if dag.out_degree(n) == 0}
    groups = {}
    for node in dag.nodes():
        if node in leaves:
            continue
        leaf_children = [c for c in dag.successors(node) if c in leaves]
        if leaf_children:
            groups[node] = leaf_children
    return groups, sorted(leaves)


def compute_cohesiveness(group_chars, char2idx, simi_matrix):
    """
    Cohesiveness of a group = minimum pairwise cosine similarity
    among its characteristics.

    Returns:
        float  – minimum similarity, or None if the group has < 2 elements
    """
    if len(group_chars) < 2:
        return None
    min_sim = float('inf')
    for c1, c2 in combinations(group_chars, 2):
        sim = simi_matrix[char2idx[c1], char2idx[c2]]
        if sim < min_sim:
            min_sim = sim
    return float(min_sim)


def find_intruders(group_chars, all_chars, min_internal_sim, char2idx, simi_matrix):
    """
    Count intruder characteristics for a group.

    An external characteristic is an intruder if its similarity with
    *any* characteristic inside the group >= the group's minimum
    internal similarity (cohesiveness).

    Returns:
        nic : int – number of intruder characteristics pairs
    """
    if min_internal_sim is None:
        return 0
    group_set = set(group_chars)
    nic = 0
    for ext in all_chars:
        if ext in group_set:
            continue
        for gc in group_chars:
            if simi_matrix[char2idx[gc], char2idx[ext]] > min_internal_sim:
                nic += 1
    return nic


def compute_group_robustness(nic, ngc, nac):
    """
    robustness_group = 1 - nic / (ngc * (nac - ngc))
    """
    denom = ngc * (nac - ngc)
    if denom == 0:
        return None
    return 1.0 - nic / denom


def compute_taxonomy_robustness(groups, all_chars, simi_matrix, cls2label=None):
    """
    Compute the overall taxonomy robustness score.

    Args:
        groups      : dict {group_node: [leaf_char, ...]}
        all_chars   : sorted list of all leaf nodes (aligned with simi_matrix)
        simi_matrix : (N_leaf, N_leaf) cosine similarity matrix
        cls2label   : optional dict for pretty-printing

    Returns:
        R_T     : float – overall robustness in [0, 1]
        details : dict  – per-group cohesiveness, intruders, robustness
    """
    char2idx = {v: i for i, v in enumerate(all_chars)}
    nac = len(all_chars)

    details = {}
    scores = []

    for gnode, gchars in tqdm(groups.items(), desc="  Computing group robustness"):
        valid = [c for c in gchars if c in char2idx]
        if len(valid) < 2:
            continue

        coh = compute_cohesiveness(valid, char2idx, simi_matrix)

        nic = find_intruders(valid, all_chars, coh, char2idx, simi_matrix)

        ngc = len(valid)
        rob = compute_group_robustness(nic, ngc, nac)

        label = cls2label.get(gnode, gnode) if cls2label else gnode
        details[gnode] = {
            'label': label,
            'ngc': ngc,
            'cohesiveness': coh,
            'intruders': nic,
            'robustness': rob,
        }
        if rob is not None:
            scores.append(rob)

    R_T = float(np.mean(scores)) if scores else 0.0
    return R_T, details


# ──────────────────────────────────────────────
#  Sampled robustness (memory-efficient for large taxonomies)
# ──────────────────────────────────────────────

def compute_cohesiveness_emb(g_idx, emb_matrix, chunk_size=2000):
    """
    Compute exact cohesiveness (min pairwise cosine similarity) from embeddings.

    For small groups the full (ngc, ngc) matrix is built at once.
    For large groups, rows are processed in chunks against the full group
    embeddings, tracking a running minimum so the complete (ngc, ngc)
    matrix is never materialised.

    Memory per chunk: chunk_size * ngc * 4 bytes.

    Args:
        g_idx      : 1-D int array of row indices into emb_matrix
        emb_matrix : (N, d) L2-normalised embeddings
        chunk_size : number of rows per chunk

    Returns:
        coh : float – exact minimum pairwise cosine similarity,
              or None if the group has < 2 elements
    """
    ngc = len(g_idx)
    if ngc < 2:
        return None

    g_emb = emb_matrix[g_idx]                        # (ngc, d)

    if ngc <= chunk_size:
        g_sim = g_emb @ g_emb.T                      # (ngc, ngc)
        np.fill_diagonal(g_sim, np.inf)
        return float(g_sim.min())

    global_min = np.inf
    for start in range(0, ngc, chunk_size):
        end = min(start + chunk_size, ngc)
        chunk_sim = g_emb[start:end] @ g_emb.T       # (chunk, ngc)
        for local_r, global_r in enumerate(range(start, end)):
            chunk_sim[local_r, global_r] = np.inf
        chunk_min = float(chunk_sim.min())
        if chunk_min < global_min:
            global_min = chunk_min
    return global_min


def compute_taxonomy_robustness_sampled(
    dag, emb_matrix, char_nodes, cls2label=None,
    s=10000, m=10, seed=42,
):
    """
    Memory-efficient robustness via sampled intruder detection.

    Cohesiveness is computed exactly once per group using chunked
    matrix multiplication (never materialises the full ngc×ngc matrix).
    Intruder detection samples `s` external characteristics per group
    and scales the count to the full external set.
    The intruder procedure is repeated `m` rounds; the final score
    is the mean.

    Memory: O(nac * d) for embeddings + O(chunk * ngc) for cohesiveness
            + O(ngc * s) for intruder detection.

    Args:
        dag        : nx.DiGraph – taxonomy (parent → child edges)
        emb_matrix : (nac, d) L2-normalised embeddings for all leaf nodes
        char_nodes : list of leaf node IDs aligned with emb_matrix rows
        cls2label  : optional dict for display
        s          : external chars to sample per group per round
        m          : number of sampling rounds
        seed       : base random seed

    Returns:
        R_T_mean : float – mean robustness across rounds
        R_T_std  : float – std of robustness across rounds
        round_scores : list[float] – per-round R(T)
        details  : dict – per-group details from the last round
    """
    groups, all_chars = extract_groups(dag)
    char2idx = {v: i for i, v in enumerate(char_nodes)}
    nac = len(all_chars)

    group_idx_map = {}
    for gnode, gchars in groups.items():
        valid = [c for c in gchars if c in char2idx]
        if len(valid) < 2:
            continue
        group_idx_map[gnode] = np.array([char2idx[c] for c in valid])

    print(f"  Sampled robustness: {len(group_idx_map)} groups with ≥2 chars, "
          f"{nac} total chars, s={s}, m={m}")

    # ── Pre-compute cohesiveness once for every group ──
    group_coh = {}
    for gnode, g_idx in tqdm(group_idx_map.items(),
                             desc="  Pre-computing cohesiveness"):
        group_coh[gnode] = compute_cohesiveness_emb(
            g_idx, emb_matrix, chunk_size=20000
        )

    # ── Repeated sampling rounds (only intruder detection varies) ──
    round_scores = []
    last_details = {}

    for r in range(m):
        rng = np.random.default_rng(seed + r)
        global_sample = rng.choice(nac, size=min(s, nac), replace=False)

        group_rob_scores = []
        details_round = {}

        for gnode, g_idx in group_idx_map.items():
            g_set = set(g_idx.tolist())
            ngc = len(g_idx)
            coh = group_coh[gnode]

            g_emb = emb_matrix[g_idx]                        # (ngc, d)

            ext_sample = np.array(
                [i for i in global_sample if i not in g_set], dtype=np.int64
            )
            if len(ext_sample) == 0:
                nic_estimated = 0.0
            else:
                ext_emb = emb_matrix[ext_sample]             # (s', d)
                cross_sim = g_emb @ ext_emb.T                # (ngc, s')
                # max_sim_per_ext = cross_sim.max(axis=0)      # (s',)
                # nic_sample = int((max_sim_per_ext > coh).sum())
                nic_sample = int((cross_sim > coh).sum())
                n_ext_total = nac - ngc
                nic_estimated = nic_sample * n_ext_total / len(ext_sample)

            rob = compute_group_robustness(nic_estimated, ngc, nac)
            if rob is not None:
                group_rob_scores.append(rob)

            label = cls2label.get(gnode, gnode) if cls2label else gnode
            details_round[gnode] = {
                'label': label,
                'ngc': ngc,
                'cohesiveness': coh,
                'intruders_estimated': nic_estimated,
                'robustness': rob,
            }

        R_T = float(np.mean(group_rob_scores)) if group_rob_scores else 0.0
        round_scores.append(R_T)
        last_details = details_round
        print(f"    Round {r+1}/{m}: R(T) = {R_T:.4f}")

    R_T_mean = float(np.mean(round_scores))
    R_T_std = float(np.std(round_scores))
    print(f"  ── Robustness (mean ± std over {m} rounds): "
          f"{R_T_mean:.4f} ± {R_T_std:.4f}")

    return R_T_mean, R_T_std, round_scores, last_details


# ──────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────

if __name__ == "__main__":

    root = config.ROOT_QID
    LABELS_FILE = config.LABELS_FILE
    TAXONOMY_FILE = config.TAXONOMY_FILE
    EMB_FILE = config.EMBEDDING_PKL_FILE
    ROBUSTNESS_OUTPUT_FILE = config.ROBUSTNESS_OUTPUT_FILE

    cls2label = {}
    with open(LABELS_FILE, 'r') as f:
        for line in f:
            terms = line.strip().split('\t')
            if len(terms) > 1:
                cls2label[terms[0]] = terms[1][1:-1]

    dag = nx.DiGraph()
    with open(TAXONOMY_FILE, 'r') as f:
        for line in f:
            if line.startswith('Q'):
                child, parent = line.strip().split(',')
                dag.add_edge(parent, child)

    print(f"  Graph: {dag.number_of_nodes()} nodes, {dag.number_of_edges()} edges")
    groups, all_chars = extract_groups(dag) # all chars: all leaf nodes: list
    print(f"  Groups: {len(groups)}, Leaf characteristics: {len(all_chars)}")

    if os.path.exists(EMB_FILE):
        emb_sub = load_submatrix(EMB_FILE, all_chars)
        simi_matrix = emb_sub @ emb_sub.T
    else:
        print(f"  Computing semantic similarity for {len(all_chars)} labels ...")
        simi_matrix = compute_semantic_similarity(all_chars, cls2label)

    R_T, details = compute_taxonomy_robustness(
        groups, all_chars, simi_matrix, cls2label
    )
    print(f"\nOverall Taxonomy Robustness: {R_T:.4f}")

    valid_groups = {k: v for k, v in details.items()
                    if v['robustness'] is not None}
    sorted_groups = sorted(valid_groups.items(),
                            key=lambda x: x[1]['robustness'])

    print(f"\n  Groups with valid robustness: {len(valid_groups)}/{len(details)}")
    if sorted_groups:
        print("  Worst 3 groups:")
        for gid, info in sorted_groups[:3]:
            print(f"    {info['label']}: coh={info['cohesiveness']:.4f}, "
                    f"intruders={info['intruders']}, rob={info['robustness']:.4f}")
        print("  Best 3 groups:")
        for gid, info in sorted_groups[-3:]:
            coh_str = f"{info['cohesiveness']:.4f}" if info['cohesiveness'] is not None else "Single Node"
            print(f"    {info['label']}: coh={coh_str}, "
                    f"intruders={info['intruders']}, rob={info['robustness']:.4f}")
    print(f"\n  Overall Taxonomy Robustness: {R_T:.4f}")

    with open(ROBUSTNESS_OUTPUT_FILE, 'w') as f:
        f.write(f"Overall Taxonomy Robustness: {R_T:.4f}\n")
        for gid, info in sorted_groups:
            f.write(f"    {info['label']}: coh={info['cohesiveness']:.4f}, "
                    f"intruders={info['intruders']}, rob={info['robustness']:.4f}\n")

    # root = 'wd:Q35120'
    # emb_path = '../../data/wikidata/'
    # emb_pkl = os.path.join(emb_path, 'wiki_ori_labels_emb.pkl')

    # cls2label = {}
    # with open(os.path.join('../../data', 'wiki_taxonomy_extracted_labels.tsv'), 'r') as f_label:
    #     for line in f_label:
    #         triple = line.strip().split('\t')
    #         if len(triple) > 3:
    #             cls2label[triple[0]] = triple[2][1:-1]

    # # first compute the wikidata taxonomy robustness (sampled – too large for full matrix)
    # wiki_path = '../../data/wikidata/'
    # dag_wiki = nx.DiGraph()
    # with open(os.path.join(wiki_path, 'wiki_taxonomy.tsv'), 'r') as taxoreader:
    #     for line in taxoreader:
    #         terms = line.strip().split('\t')
    #         if len(terms) > 3:
    #             child, parent = terms[0], terms[2]
    #             dag_wiki.add_edge(parent, child)

    # _, all_chars_wiki = extract_groups(dag_wiki)
    # print(f"\n{'='*50}")
    # print(f" Wikidata taxonomy")
    # print(f"{'='*50}")
    # print(f"  Graph: {dag_wiki.number_of_nodes()} nodes, "
    #       f"{dag_wiki.number_of_edges()} edges")
    # print(f"  Leaf characteristics: {len(all_chars_wiki)}")

    # if os.path.exists(emb_pkl):
    #     emb_wiki = load_submatrix(emb_pkl, all_chars_wiki)
    #     print(f"  Loaded embeddings: {emb_wiki.shape}")
    # else:
    #     raise FileNotFoundError(
    #         f"Pre-computed embeddings not found at {emb_pkl}. "
    #         "Run embedding generation first."
    #     )

    # R_T_mean, R_T_std, round_scores, details_wiki = \
    #     compute_taxonomy_robustness_sampled(
    #         dag_wiki, emb_wiki, all_chars_wiki, cls2label,
    #         s=20000, m=10, seed=42,
    #     )
    # print(f"\n  Wikidata Robustness: {R_T_mean:.4f} ± {R_T_std:.4f}")

    # del emb_wiki, details_wiki