"""
Graph Edit Distance (GED) Algorithm

Computes GED between two directed graphs represented as sparse adjacency
matrices with named node lists, following the specification in ged.md.

Cost model: unit cost (1) for each node/edge edit operation.

C_GED = C1 + C2 + C3 + C4
  C1: node deletions   = |N| - |N ∩ N'|
  C2: edge edits on shared nodes = |A_sub - A'_sub|  (L1 norm)
  C3: node insertions   = |N'| - |N ∩ N'|
  C4: edge insertions for new nodes = |A' - A'_sub_expanded|  (L1 norm)
"""

import numpy as np
from scipy import sparse
from typing import Dict, List, Tuple, Union
import networkx as nx
import os
from tqdm import tqdm
import json
from sknetwork.path import breadth_first_search

def graph_edit_distance(
    A: sparse.spmatrix,
    N: List[str],
    A_prime: sparse.spmatrix,
    N_prime: List[str],
) -> Tuple[int, Dict[str, int]]:
    """
    Compute the Graph Edit Distance (GED) from graph G to graph G'.

    Parameters
    ----------
    A : scipy.sparse matrix, shape (n, n)
        Binary adjacency matrix of directed graph G.
    N : list of str, length n
        Node ID list of G. N[i] is the ID of the i-th row/column of A.
    A_prime : scipy.sparse matrix, shape (m, m)
        Binary adjacency matrix of directed graph G'.
    N_prime : list of str, length m
        Node ID list of G'. N_prime[i] is the ID of the i-th row/column of A'.

    Returns
    -------
    ged : int
        The total graph edit distance.
    details : dict
        Breakdown: C1 (node deletions), C1b (edge deletions for deleted nodes),
        C2 (edge edits on shared nodes), C3 (node insertions),
        C4 (edge insertions for new nodes), and n_cap (number of shared nodes).
    """
    # -------------------------------------------------------------------
    # Step 1: Node Deletion Cost
    # -------------------------------------------------------------------
    N_set = set(N)
    N_prime_set = set(N_prime)
    N_cap = sorted(N_set & N_prime_set)  # sorted for deterministic ordering
    n_cap = len(N_cap)

    C1 = len(N) - n_cap  # nodes in G but not in G'

    # -------------------------------------------------------------------
    # Step 2: Extract Subgraphs on Intersected Nodes
    # -------------------------------------------------------------------
    # Build index lookups: node ID -> position in the original matrix
    node_to_idx = {node: i for i, node in enumerate(N)}
    node_to_idx_prime = {node: i for i, node in enumerate(N_prime)}

    # Indices of intersected nodes in A and A' (aligned by N_cap ordering)
    idx_in_A = [node_to_idx[node] for node in N_cap]
    idx_in_A_prime = [node_to_idx_prime[node] for node in N_cap]

    # Convert to CSR for efficient row/column slicing
    A_csr = sparse.csr_matrix(A)
    A_prime_csr = sparse.csr_matrix(A_prime)

    if n_cap > 0:
        # A_sub: submatrix of A restricted to N_cap (n_cap x n_cap)
        A_sub = A_csr[np.ix_(idx_in_A, idx_in_A)]
        # A'_sub: submatrix of A' restricted to N_cap (n_cap x n_cap)
        A_prime_sub = A_prime_csr[np.ix_(idx_in_A_prime, idx_in_A_prime)]
    else:
        print("n_cap is 0, A_sub and A_prime_sub are empty")
        A_sub = sparse.csr_matrix((0, 0))
        A_prime_sub = sparse.csr_matrix((0, 0))

    # -------------------------------------------------------------------
    # Step 3: Edge Deletion Cost for Deleted Nodes
    # -------------------------------------------------------------------
    # Edges in G where at least one endpoint was deleted (not in G').
    # This equals |A| - |A_sub|, mirroring C4 for insertions.
    total_edges_A = int(abs(A_csr).sum())
    if n_cap > 0:
        edges_in_A_sub = int(abs(A_sub).sum())
    else:
        edges_in_A_sub = 0
    C1b = total_edges_A - edges_in_A_sub

    # -------------------------------------------------------------------
    # Step 4: Edge Edit Cost on Intersected Nodes
    # -------------------------------------------------------------------
    # Both A_sub and A'_sub are aligned to the same N_cap ordering,
    # so element-wise difference is meaningful.
    if n_cap > 0:
        C2 = int(abs(A_sub - A_prime_sub).sum())
    else:
        C2 = 0

    # -------------------------------------------------------------------
    # Step 5: Node Insertion Cost
    # -------------------------------------------------------------------
    C3 = len(N_prime) - n_cap  # nodes in G' but not in G

    # -------------------------------------------------------------------
    # Step 6: Edge Insertion Cost for New Nodes
    # -------------------------------------------------------------------
    # A'_sub_expanded is an (m x m) matrix that equals A' at positions
    # (i, j) where both i and j correspond to N_cap nodes, and 0 elsewhere.
    # Therefore:
    #   |A' - A'_sub_expanded| = |A'| - |A'_sub|
    # because A'_sub is exactly the restriction of A' to N_cap rows/cols.
    total_edges_A_prime = int(abs(A_prime_csr).sum())
    if n_cap > 0:
        edges_in_A_prime_sub = int(abs(A_prime_sub).sum())
    else:
        edges_in_A_prime_sub = 0

    C4 = total_edges_A_prime - edges_in_A_prime_sub

    # -------------------------------------------------------------------
    # Final GED
    # -------------------------------------------------------------------
    ged = C1 + C1b + C2 + C3 + C4

    details = {
        "C1_node_deletions": C1,
        "C1b_edge_deletions": C1b,
        "C2_edge_edits_shared": C2,
        "C3_node_additions": C3,
        "C4_edge_insertions_new": C4,
        "n_cap": n_cap,
        "ged": ged,
    }

    return ged, details


# ======================================================================
# Helper utilities
# ======================================================================


def digraph_to_sparse(
    edges: List[Tuple[str, str]],
) -> Tuple[sparse.csr_matrix, List[str]]:
    """
    Convert an edge list of a directed graph to a sparse adjacency matrix.

    Parameters
    ----------
    edges : list of (parent, child) tuples
        Each tuple represents a directed edge parent -> child.

    Returns
    -------
    A : scipy.sparse.csr_matrix, shape (n, n)
        Binary adjacency matrix.
    N : list of str
        Sorted node ID list. N[i] is the node corresponding to row/col i.
    """
    nodes = set()
    for u, v in edges:
        nodes.add(u)
        nodes.add(v)
    N = sorted(nodes)
    node_to_idx = {node: i for i, node in enumerate(N)}

    rows, cols = [], []
    for u, v in edges:
        rows.append(node_to_idx[u])
        cols.append(node_to_idx[v])

    n = len(N)
    data = np.ones(len(rows), dtype=np.int8)
    A = sparse.csr_matrix((data, (rows, cols)), shape=(n, n))
    return A, N


def nx_to_sparse(
    G,
) -> Tuple[sparse.csr_matrix, List[str]]:
    """
    Convert a NetworkX DiGraph to a sparse adjacency matrix with a node list.

    Parameters
    ----------
    G : networkx.DiGraph
        A directed graph.

    Returns
    -------
    A : scipy.sparse.csr_matrix, shape (n, n)
        Binary adjacency matrix.
    N : list of str
        Sorted node ID list.
    """
    N = sorted(G.nodes())
    node_to_idx = {node: i for i, node in enumerate(N)}
    rows, cols = [], []
    for u, v in G.edges():
        rows.append(node_to_idx[u])
        cols.append(node_to_idx[v])
    n = len(N)
    data = np.ones(len(rows), dtype=np.int8)
    A = sparse.csr_matrix((data, (rows, cols)), shape=(n, n))
    return A, N

# ======================================================================
# NetworkX GED
# ======================================================================
def run_nx_ged(G, G_prime):
    """
    Compute networkx graph edit distance, making node identity matter.

    By default, networkx.graph_edit_distance allows any node relabeling, which can give GED=0
    for graphs like [('a','b')] and [('c','d')] because it just maps 'a'->'c', 'b'->'d'.

    To force networkx to only allow matching nodes with the same label (i.e., 'a' can only be matched to 'a'),
    provide a node_match function that checks equality.

    NOTE: For this to take effect, nodes must have some attribute, or node_match will never be called!
    So, add a 'label' attribute with their name, and match on 'label'.

    See: https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.similarity.graph_edit_distance.html

    Parameters
    ----------
    G : nx.Graph or nx.DiGraph
    G_prime : nx.Graph or nx.DiGraph

    Returns
    -------
    ged : float
        The graph edit distance.
    """

    # Patch: Give each node a 'label' attribute with its node name
    G_labeled = G.copy()
    G_prime_labeled = G_prime.copy()
    for n in G_labeled.nodes:
        G_labeled.nodes[n]['label'] = n
    for n in G_prime_labeled.nodes:
        G_prime_labeled.nodes[n]['label'] = n

    # Node match only if labels are identical
    def node_match(n1_attrs, n2_attrs):
        return n1_attrs.get('label') == n2_attrs.get('label')

    ged = nx.graph_edit_distance(
        G_labeled, G_prime_labeled,
        node_match=node_match
    )
    return ged


def get_paths_from_root(adjacency, node_names, target):
    """
    Extract the subgraph and node list corresponding to all ancestors of a given target node.

    Parameters
    ----------
    adjacency : scipy.sparse matrix
        Adjacency matrix (interpreted as parent -> child edges).
    node_names : list of str
        Node names corresponding to rows/columns of adjacency.
    target : str
        The node for which to extract the ancestors' subgraph.

    Returns
    -------
    extract_adj : scipy.sparse matrix
        Submatrix of adjacency containing only the ancestor nodes.
    extract_nodes : list of str
        Names of the nodes corresponding to extract_adj.
    """
    # Find the index of the target node in node_names
    end_index = node_names.index(target)
    # Transpose adjacency: treat edges as subclass_of pointing to parents
    adjacency_transpose = adjacency.T
    # Use breadth-first search to get all ancestors (including the target)
    ancestors = breadth_first_search(adjacency_transpose, source=end_index)
    # Extract the submatrix and node names for the ancestors
    extract_adj = adjacency_transpose[ancestors, :][:, ancestors]
    extract_nodes = list(np.array(node_names)[ancestors])
    return extract_adj, extract_nodes


def get_subgraph_from_root(digraph, root):
    # get all descendants of the given root
    descendants = nx.descendants(digraph, root)
    descendants.add(root)
    # get the subgraph
    subgraph = digraph.subgraph(descendants)
    # check stats
    if not nx.is_directed_acyclic_graph(subgraph):
        raise ValueError(f"Subgraph from {root} is not a directed acyclic graph.")
    if not nx.is_weakly_connected(subgraph):
        raise ValueError(f"Subgraph from {root} is not weakly connected.")
    num_roots = len([n for n,d in subgraph.in_degree() if d==0])
    if num_roots != 1:
        raise ValueError(f"Subgraph from {root} has {num_roots} root nodes.")
    return subgraph

# ======================================================================
# Self-test / Example
# ======================================================================

if __name__ == "__main__":
    models = ['gemma4b', 'gemma12b', 'gemma27b', 'qwen8b', 'qwen14b', 'qwen32b', 'mistral7b', 'mistral24b', 'mixtral8x7b', 'llama8b']
    data_path = '../../data/wikidata/' 
    wiki_path_2026 = '../../results/wikc_in_llms/wikidata/Version-2026/'
    wiki_path_2024 = '../../results/wikc_in_llms/wikidata/Version-2024/'
    save_path = '../../results/ged_new/res/'
    
        
    # ================================
    # 1. Comparision between different models'sub-taxonomies
    # ================================
    print("Comparision between different models'sub-taxonomies")
    roots = ['wd:Q35120']
    roots = roots + ['wd:Q215627', 'wd:Q17537576', 'wd:Q43229', 'wd:Q1656682','wd:Q3622002'] # person, creative work, organization, event, geo_area.
    roots = roots + ['wd:Q7930989', 'wd:Q6256', 'wd:Q27096213'] # city, country, geographical entity
    roots = roots + ['wd:Q783794', 'wd:Q5341295', 'wd:Q327055', 'wd:Q11424'] # company, education orgnization, worker, film
    roots = roots + ['wd:Q483501', 'wd:Q386724'] # artist, work
    roots = roots + ['wd:Q3257686', 'wd:Q56061', 'wd:Q27096235'] 
    # locality, administraitive territorial entity: country -> territory entity, artificial geographic entity: city -> artificial geographic entity

    # roots = ['wd:Q35120'] # entity as root
    for root in tqdm(roots):
        root_matrix = np.zeros((len(models), len(models)))
        for model in models:
            GRAPH_PATH1 = f'../../../results/wikc_in_llms/{model}/'
            root_model1 = nx.DiGraph()
            # with open(os.path.join(GRAPH_PATH1,f'html/{model}_wikc_subtree_from_{root}.txt'), 'r') as taxoreader:
            with open(os.path.join(GRAPH_PATH1, 'wikc_new_add_city.txt'), 'r') as taxoreader:
                for line in taxoreader:
                    child, parent = line.strip().split('\t')
                    root_model1.add_edge(parent, child)
            # check if root exists
            if not root_model1.has_node(root):
                print(f"Root {root} not in the {model} graph.")
                continue
            root_model1 = get_subgraph_from_root(root_model1, root=root)
            # turn to sparse matrix and node list
            A_root_model1, N_root_model1 = nx_to_sparse(root_model1)
            
            for model2 in models:
                if model == model2:
                    root_matrix[(models.index(model), models.index(model2))] = 0
                    continue
                GRAPH_PATH2 = f'../../../results/wikc_in_llms/{model2}/'
                root_model2 = nx.DiGraph()
                with open(os.path.join(GRAPH_PATH2, 'wikc_new_add_city.txt'), 'r') as taxoreader:
                # with open(os.path.join(GRAPH_PATH2,f'html/{model2}_wikc_subtree_from_{root}.txt'), 'r') as taxoreader:
                    for line in taxoreader:
                        child, parent = line.strip().split('\t')
                        root_model2.add_edge(parent, child)
                # check if root exists
                if not root_model2.has_node(root):
                    print(f"Root {root} not in the {model2} graph.")
                    continue
                root_model2 = get_subgraph_from_root(root_model2, root=root)
                # turn to sparse matrix and node list
                A_root_model2, N_root_model2 = nx_to_sparse(root_model2)

                # graph edit distance
                ged, details = graph_edit_distance(A_root_model1, N_root_model1, A_root_model2, N_root_model2)
                root_matrix[(models.index(model), models.index(model2))] = details["ged"]
        # store the matrix
        np.save(f'res/ged_matrix_rootsubtree_{root[3:]}.npy', root_matrix)
