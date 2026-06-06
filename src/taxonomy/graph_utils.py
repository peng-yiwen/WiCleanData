from collections import deque
import warnings
warnings.filterwarnings("ignore")
from sknetwork.path import get_distances, breadth_first_search
from sknetwork.visualization import svg_graph
from IPython.display import SVG, display
from collections import defaultdict
import networkx as nx
from scipy.sparse import csr_matrix
import numpy as np
import torch
import torch.nn.functional as F
from itertools import groupby
from tqdm import tqdm
from torch import Tensor
import pickle
import os


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


def getSuperClasses(cls, classes, WikiTaxonomyUp):
    """Adds all superclasses of a class <cls> (including <cls>) to the set <classes>"""
    classes.add(cls)
    # Make a check before because it's a defaultdict,
    # which would create cls if it's not there
    if cls in WikiTaxonomyUp:
        for sc in WikiTaxonomyUp[cls]:
            getSuperClasses(sc, classes, WikiTaxonomyUp)      


def getAncestors(cls, WikiTaxonomyUp):
    """Returns the set of all parent classes of <cls> (including <cls>!)"""
    classes=set()
    getSuperClasses(cls, classes, WikiTaxonomyUp)        
    return classes


def cumulative_stats(stats, TaxonomyUp):
    """Cumulative statistics of classes"""
    cum_stats = defaultdict(int)
    for instantiated_cls in stats.keys():
        ancestors = getAncestors(instantiated_cls, TaxonomyUp) # including cls itself
        for ancestor in ancestors:
            cum_stats[ancestor] += stats[instantiated_cls]
    return cum_stats


def get_mapped_ancestors(digraph, ancestors, cls, mapped_wiki_ents):
    for sp in digraph.predecessors(cls):
        # the condition to rebuild
        if sp in mapped_wiki_ents: 
            ancestors.add(sp)
            continue
        get_mapped_ancestors(digraph, ancestors, sp, mapped_wiki_ents)


def get_first_ancestors_for_rebuild(digraph, cls, wikipedia_ents):
    ancestors = set()
    get_mapped_ancestors(digraph, ancestors, cls, wikipedia_ents)
    return ancestors


def load_wikipedia_mapped_ents(graph, loc):
    """ 
    Loads the mapped Wikipedia entities into Wikidata.
    """
    mapped_wiki_ents = set()
    wikipedia_lists = ['enwiki', 'frwiki', 'dewiki', 'zhwiki', 'arwiki', 'eswiki'] # 5 different lanugages
    for wikifile in wikipedia_lists:
        with open(os.path.join(loc, wikifile), 'r') as file:
            for line in file:
                qid = line.strip().split(',')[1]
                prefix_qid = 'wd:'+str(qid)
                if graph.has_node(prefix_qid):
                    mapped_wiki_ents.add(prefix_qid)
    return mapped_wiki_ents


def cumulative_stats_for_class(cls, stats, taxonomyDown):
    """Cumulative statistics of classes
    Args:
        cls (str): class to be calculated.
        stats (dict): dict of direct instances count for each class.
        taxonomyDown (dict): Taxonomy from top to down.
    """
    descendants = getDescendants(cls, taxonomyDown) # including cls itself
    return sum(stats.get(descendant, 0) for descendant in descendants)


def bfs_edges_by_level(graph, root):
    '''
    BFS traversal of edges according to its depth level
    @param graph: directed acyclic graph (taxonomy)
    @param root: root node of the graph
    '''
    visited = set()
    queue = deque([(root, 0)])
    # dictionary to hold edges by their depth level
    edges_by_level = {}

    while queue:
        node, depth = queue.popleft()
        if node not in visited:
            visited.add(node)
            for neighbor in graph.neighbors(node):
                edge = (node, neighbor) # parent-child edge
                if depth not in edges_by_level:
                    edges_by_level[depth] = []
                edges_by_level[depth].append(edge)
                if neighbor not in visited:
                    queue.append((neighbor, depth + 1)) # same node may have different levels
    
    # sort the edges per level by in_degree of children (descending)
    bfs_edges = []
    for depth in sorted(edges_by_level.keys()):
        edge_list = edges_by_level[depth]
        # sort by in_degree of child (edge[1]), descending
        edge_list.sort(key=lambda item: (graph.in_degree(item[1]), item[1], item[0]), reverse=True)
        bfs_edges.extend(edge_list)

    return bfs_edges


def reorder_edges_by_similarity(graph, bfs_edges, simi_matrix, node2id, reverse=False):
    reorder_edges = []
    for key, group in groupby(bfs_edges, key=lambda x: (graph.in_degree(x[1]), x[1])):
        group_list = list(group)
        if len(group_list) > 1:
            if reverse: # descending order
                group_list.sort(key=lambda edge: -simi_matrix[
                    node2id[edge[0]], 
                    node2id[edge[1]]
                    ])
            else: # ascending order
                group_list.sort(key=lambda edge: simi_matrix[
                node2id[edge[0]], 
                node2id[edge[1]]
                ])
        reorder_edges.extend(group_list)
    return reorder_edges
   

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


def draw_graph(graph, target, cls2label, mapping=None):
    '''
    Draw the graph from the target node to the root.
    @param: cls2label: dict of class to its label
    @param: mapping: dict of Wikidata entity to WiKC entity
    '''
    # Get valid target
    while not graph.has_node(target):
        if target not in mapping: # may delete this node by cutting
            print(f"Node {target} not in the graph.")
            return
        target = mapping[target]
        
    adjacency = csr_matrix(nx.adjacency_matrix(graph).toarray())
    names = list(graph.nodes)
    end_index = names.index(target)
    adjacency_transpose = adjacency.T # subclass_of relation
    ancestors = breadth_first_search(adjacency_transpose, source=end_index)
    extract_ = adjacency_transpose[ancestors, :][:, ancestors]
    extract_nodes = list(np.array(names)[ancestors])

    root_index = extract_nodes.index('wd:Q35120')
    distances = get_distances(extract_.T, source=root_index)
    extract_names = [cls2label[qid] for qid in extract_nodes]
    weights = np.clip(~distances+max(distances) - 1, a_min=0, a_max=3) 
    image = svg_graph(extract_, names=extract_names, display_node_weight=True, 
                    node_weights=weights, node_size_max=12, node_size_min=3,
                    scores=-distances, scale=1, font_size=8)
    # Display the SVG image
    svg_image = SVG(image)
    display(svg_image)
    return image
    # # Save the SVG image to a file
    # with open(f"{target[3:]}.svg", "w") as svg_file:
    #     svg_file.write(image)



def save_graph_checkpoint(graph, path, filename):
    with open(path+filename, 'w') as taxowriter:
        for edge in graph.edges():
            parent, child = edge
            taxowriter.write(child+'\t'+parent+'\n')


def load_graph_checkpoint(path, filename):
    graph = nx.DiGraph()
    with open(path+filename, 'r') as taxoreader:
        for line in taxoreader:
            child, parent = line.strip().split('\t')
            graph.add_edge(parent, child)
    return graph


def format_taxonomy(path, digraph):
    '''
    Format the taxonomy for nt version.
    '''
    with open(path+'WiKC.nt', 'w') as taxowriter:
        for edge in digraph.edges():
            parent, child = edge
            formated_child = '<http://www.wikidata.org/entity/'+child[3:]+'>'
            formated_parent = '<http://www.wikidata.org/entity/'+parent[3:]+'>'
            rel = '<http://www.wikidata.org/prop/direct/P279>'
            taxowriter.write(formated_child+' '+rel+' '+formated_parent+' .\n')


def generate_html(node, taxonomy, cls2label):
    html = '<ul>'
    for child in taxonomy.get(node, []):
        html += f'<li><span class="toggle" onclick="toggleChildren(this)">&#9660;</span>{cls2label[child]}({child[3:]})<ul class="children">'
        html += generate_html(child, taxonomy, cls2label)
        html += '</ul></li>'
    html += '</ul>'
    return html


def visualize_taxonomy_by_html(root, wikiTaxonDown, cls2label, loc):
    '''
    Visualize the taxonomy in HTML format.
    '''
    # Create the HTML content
    html_content = generate_html(root, wikiTaxonDown, cls2label)

    # Generate the complete HTML file
    html_template = f'''
    <!DOCTYPE html> 
    <html>
    <head>
        <style>
            ul {{
                list-style-type: none;
            }}
            li {{
                padding-left: 10px;
            }}
            .toggle {{
                cursor: pointer;
                color: black;
            }}
            .children {{
                display: none;
            }}
        </style>
        <script>
            function toggleChildren(element) {{
                var ul = element.nextElementSibling;
                if (ul.style.display === 'none' || ul.style.display === '') {{
                    ul.style.display = 'block';
                    element.textContent = '▶';
                    element.style.color = 'blue';
                }} else {{
                    ul.style.display = 'none';
                    element.textContent = '▼';
                    element.style.color = 'black';
                }}
            }}
        </script>
    </head>
    <body>
        <h1>Wikidata 2026.01 Taxonomy: {cls2label[root]}</h1>
        {html_content}
    </body>
    </html>
    '''

    # Write the HTML content to a file
    with open(os.path.join(loc, f"{cls2label[root]}.html"), "w") as html_file:
        html_file.write(html_template)

    print(f"HTML file generated: {cls2label[root]}.html")


def reachable_leaves(graph, source):
    if source not in graph:
        return set()
    leaves = set()
    visited = set()
    queue = deque([source])

    while queue:
        node = queue.popleft()
        if node in visited:
            continue
        visited.add(node)
        neighbors = list(graph.successors(node))
        if not neighbors:  # out-degree 0
            leaves.add(node)
        else:
            queue.extend(neighbors)
    return leaves

#### model
def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def encode_text(model, tokenizer, input_texts, device=None):
    # Tokenize the input texts
    batch_dict = tokenizer(input_texts, max_length=512, padding=True, truncation=True, return_tensors='pt').to(device)
    outputs = model(**batch_dict)
    embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
    embeddings = F.normalize(embeddings, p=2, dim=1)

    return embeddings


def calculate_simi_matrix(graph, model, tokenizer, cls2label, path):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    nodes = sorted(list(graph.nodes()))
    node2id = {node: i for i, node in enumerate(nodes)}
    labels = [cls2label[node] for node in nodes]
    adjacency_matrix = nx.to_numpy_array(graph, nodelist=nodes)
    embeddings = []
    for i in tqdm(range(0, len(labels), 128), desc="    Calculating class embeddings"):
        batch_labels = labels[i:i + 128]
        with torch.no_grad():
            emb = encode_text(model, tokenizer, batch_labels, device=device).cpu()
            embeddings.append(emb)
    # Stack embeddings to form the matrix
    embedding_matrix = torch.cat(embeddings, dim=0)
    embedding_matrix = embedding_matrix.numpy()
    # calculate cosine similarity based on adjacency
    simi_matrix = embedding_matrix @ embedding_matrix.T
    simi_matrix = simi_matrix * adjacency_matrix  # mask non-adjacent pairs
    # goes to sparse matrix
    simi_matrix = csr_matrix(simi_matrix)

    # store the similarity matrix
    result = {"id": node2id, "simi": simi_matrix}
    with open(os.path.join(path, 'simi_matrix.pkl'), 'wb') as f:
        pickle.dump(result, f)
    return simi_matrix, node2id


########################################################
# Used for intrinsic analysis
########################################################


def count_paths_to_root(digraph_reverse, root, nodes_list):
    """
    Count the number of paths from each node to the root
    Args:
        digraph_reverse (nx.DiGraph): The directed acyclic graph. (directetion: child to parent)
        root (str): The root node.
        nodes_list (list): The list of nodes to count the paths to the root.
    Returns:
        path_counts (dict): A dictionary of node to the number of paths to the root.
    """
    path_counts = {}
    for node in nodes_list:
        if node == root:
            path_counts[node] = 0  # Only one path from root to itself
        else:
            paths = list(nx.all_simple_paths(digraph_reverse, node, root)) # paths from node to root
            path_counts[node] = len(paths)
    return path_counts


def avg_path_to_root(digraph, cls_inst_stats, root):
    """
    Calculate the average number of paths to the root (w.r.t. instances)
    Args:
        cls_inst_stats (dict): dict of class to its direct instance count
    """
    cls_list = [cls for cls in cls_inst_stats.keys() if digraph.has_node(cls) and cls_inst_stats.get(cls, 0) > 0]
    # insts = sum([cls_inst_stats[cls] for cls in cls_list])
    path_counts = count_paths_to_root(digraph.reverse(), root, cls_list)
    # avg number of paths to root
    total_inst_path_count = 0
    total_inst_count = 0
    for cls, count in path_counts.items():
        total_inst_path_count += count * cls_inst_stats[cls]
        total_inst_count += cls_inst_stats[cls]
    return total_inst_path_count / total_inst_count