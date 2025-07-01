import os
import ast
from difflib import SequenceMatcher
import tokenize
from io import BytesIO
from sklearn.cluster import DBSCAN
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import graphviz
import seaborn as sns
from pathlib import Path
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend for Flask
import matplotlib.pyplot as plt

# Similarity functions (AST, token, and control flow)
def ast_similarity(code1, code2):
    try:
        tree1, tree2 = ast.parse(code1), ast.parse(code2)
    except SyntaxError:
        return 0.0
    nodes1 = [type(node).__name__ for node in ast.walk(tree1)]
    nodes2 = [type(node).__name__ for node in ast.walk(tree2)]
    return SequenceMatcher(None, nodes1, nodes2).ratio()

def tokenize_code(code):
    tokens = []
    g = tokenize.tokenize(BytesIO(code.encode('utf-8')).readline)
    for token in g:
        if token.type == tokenize.NAME:
            tokens.append('IDENTIFIER')
        elif token.type == tokenize.OP:
            tokens.append(token.string)
        elif token.type == tokenize.NUMBER:
            tokens.append('NUMBER')
        elif token.type == tokenize.STRING:
            tokens.append('STRING')
    return tokens

def token_similarity(code1, code2):
    tokens1, tokens2 = tokenize_code(code1), tokenize_code(code2)
    set1 = set(tokens1)
    set2 = set(tokens2)
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    return len(intersection) / len(union) if len(union) != 0 else 1.0

from networkx.algorithms.similarity import graph_edit_distance

# Function to build a control flow graph from code
class ControlFlowGraphBuilder(ast.NodeVisitor):
    def __init__(self):
        self.cfg = nx.DiGraph()
        self.current_node = None
        self.node_count = 0

    def add_node(self, label):
        node_id = self.node_count
        self.cfg.add_node(node_id, label=label)
        if self.current_node is not None:
            self.cfg.add_edge(self.current_node, node_id)
        self.current_node = node_id
        self.node_count += 1

    def visit_FunctionDef(self, node):
        self.add_node(f"FunctionDef: {node.name}")
        self.generic_visit(node)

    def visit_If(self, node):
        self.add_node("If")
        self.generic_visit(node)

    def visit_For(self, node):
        self.add_node("For")
        self.generic_visit(node)

    def visit_While(self, node):
        self.add_node("While")
        self.generic_visit(node)

    def visit_Return(self, node):
        self.add_node("Return")

def build_control_flow_graph(code):
    tree = ast.parse(code)
    builder = ControlFlowGraphBuilder()
    builder.visit(tree)
    return builder.cfg

def control_flow_similarity(code1, code2):
    cfg1 = build_control_flow_graph(code1)
    cfg2 = build_control_flow_graph(code2)
    ged = graph_edit_distance(cfg1, cfg2)
    normalized_score = 1 / (1 + ged) if ged is not None else 0
    return normalized_score

def calculate_weighted_similarity(ast_sim, token_sim, control_flow_sim, weights=(0.4, 0.4, 0.2)):
    return (weights[0] * ast_sim) + (weights[1] * token_sim) + (weights[2] * control_flow_sim)

def load_codes_from_folder(folder_path):
    code_files = [f for f in os.listdir(folder_path) if f.endswith(".py")]
    codes = {}
    for file in code_files:
        with open(os.path.join(folder_path, file), 'r') as f:
            codes[file] = f.read()
    return codes

def calculate_pairwise_similarities(codes):
    similarity_threshold = 0.7
    files = list(codes.keys())
    similarities = []
    for i, file1 in enumerate(files):
        for j, file2 in enumerate(files):
            if i < j:
                ast_sim = ast_similarity(codes[file1], codes[file2])
                token_sim = token_similarity(codes[file1], codes[file2])
                control_flow_sim = control_flow_similarity(codes[file1], codes[file2])
                overall_sim = calculate_weighted_similarity(ast_sim, token_sim, control_flow_sim)
                if overall_sim > similarity_threshold:
                    similarities.append((file1, file2, overall_sim))
    return similarities

def cluster_codes(similarities, codes, eps=0.5, min_samples=2):
    files = list(codes.keys())
    n = len(files)
    similarity_matrix = np.zeros((n, n))
    for (file1, file2, sim) in similarities:
        i, j = files.index(file1), files.index(file2)
        similarity_matrix[i][j] = similarity_matrix[j][i] = sim
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed")
    labels = clustering.fit_predict(1 - similarity_matrix)
    clusters = {}
    for idx, label in enumerate(labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(files[idx])
    # save_similarity_to_csv(similarity_matrix, files, output_path="similarity_matrix.csv")
    return clusters, similarity_matrix, files

def save_similarity_to_csv(similarity_matrix, files, output_path="similarity_matrix.csv"):
    import pandas as pd
    df = pd.DataFrame(similarity_matrix, index=files, columns=files).round(2)
    df.to_csv(output_path)
    print(f"CSV file saved to {output_path}")

def set_cell_font(cell, bold=False, font_name='Calibri', font_size=10):
    paragraphs = cell.paragraphs
    for paragraph in paragraphs:
        run = paragraph.runs[0] if paragraph.runs else paragraph.add_run()
        run.font.name = font_name
        run.font.size = Pt(font_size)
        run.bold = bold

def set_cell_width(cell, width_in_inches=1.5):
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    tcW = OxmlElement('w:tcW')
    tcW.set(qn('w:type'), 'dxa')
    tcW.set(qn('w:w'), str(int(width_in_inches * 1440)))  # 1 inch = 1440 twips
    tcPr.append(tcW)






# Visualization functions
def visualize_similarity_matrix(similarity_matrix, files, output_dir):
    # Ensure diagonal values are 1
    np.fill_diagonal(similarity_matrix, 1)
    

    # Set figure size and create the heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        similarity_matrix, 
        annot=True, 
        fmt=".2f", 
        xticklabels=files, 
        yticklabels=files, 
        cmap="YlGnBu", 
        cbar=True, 
        square=True  # Ensures cells are square-shaped
    )
    
    # Rotate labels for better readability
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    
    # Add a title and adjust layout
    plt.title("Code Similarity Heatmap", fontsize=16, pad=20)
    plt.tight_layout()  # Automatically adjust layout to prevent cutoff

    # Save the heatmap
    filename = os.path.join(output_dir, "similarity_matrix.png")
    plt.savefig(filename, dpi=300)  # High resolution for better quality
    plt.close()
    
    return filename

def visualize_ast(code, filename):
    try:
        tree = ast.parse(code)
        graph = nx.DiGraph()  # Directed Graph to maintain tree structure

        def add_nodes_edges(node, parent=None):
            node_id = str(id(node))
            graph.add_node(node_id, label=type(node).__name__)

            if parent:
                graph.add_edge(parent, node_id)

            for child in ast.iter_child_nodes(node):
                add_nodes_edges(child, node_id)

        add_nodes_edges(tree)

        # Extract labels for nodes
        labels = nx.get_node_attributes(graph, 'label')

        # Use `spring_layout` for a simple tree-like structure
        pos = nx.spring_layout(graph, seed=42)

        # Plot the tree
        plt.figure(figsize=(12, 6))
        nx.draw(graph, pos, labels=labels, with_labels=True, node_color="lightblue",
                edge_color="gray", node_size=3000, font_size=10, font_weight="bold")

        # Save the file
        plt.savefig(filename, format='png', dpi=300)
        plt.close()

        return filename

    except SyntaxError:
        return None

def visualize_tokens(code, filename):
    tokens = tokenize_code(code)
    token_counts = {token: tokens.count(token) for token in set(tokens)}
    
    plt.figure(figsize=(10, 5))
    plt.bar(token_counts.keys(), token_counts.values(), color='skyblue')
    plt.title("Token Frequency")
    plt.xlabel("Token Type")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    filepath = f"{filename}.png"
    plt.savefig(filepath)
    plt.close()
    return filepath

def visualize_control_flow(code, filename):
    try:
        tree = ast.parse(code)
        graph = nx.DiGraph()

        def add_nodes_edges(node, parent=None):
            node_id = str(id(node))
            graph.add_node(node_id, label=type(node).__name__)
            if parent:
                graph.add_edge(str(id(parent)), node_id)
            for child in ast.iter_child_nodes(node):
                add_nodes_edges(child, node)

        add_nodes_edges(tree)

        pos = nx.spring_layout(graph, k=0.5)  # Adjust k to increase spacing
        labels = nx.get_node_attributes(graph, 'label')
        node_sizes = [1000 if labels[node] == 'FunctionDef' else 300 for node in graph]
        
        plt.figure(figsize=(12, 8))
        nx.draw(graph, pos, labels=labels, with_labels=True, node_size=node_sizes, node_color="lightblue", font_size=10)
        
        filepath = f"{filename}.png"
        plt.savefig(filepath)
        plt.close()
        return filepath
    except SyntaxError:
        return None


def visualize_similarity_graph(similarity_matrix, files, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    G = nx.Graph()

    for file in files:
        G.add_node(file)

    similarity_values = []
    edges = []
    for i in range(len(files)):
        for j in range(i + 1, len(files)):
            similarity_score = similarity_matrix[i][j]
            if similarity_score > 0:
                G.add_edge(files[i], files[j], weight=similarity_score)
                similarity_values.append(similarity_score)
                edges.append((files[i], files[j]))

    min_sim = min(similarity_values) if similarity_values else 0
    max_sim = max(similarity_values) if similarity_values else 1
    edge_colors = [
        (similarity_matrix[i][j] - min_sim) / (max_sim - min_sim)
        for i in range(len(files)) for j in range(i + 1, len(files))
        if similarity_matrix[i][j] > 0
    ]

    pos = nx.spring_layout(G, seed=42, k=1.5, iterations=50)

    fig, ax = plt.subplots(figsize=(12, 8))
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=700, node_color="lightblue", edgecolors="black")
    nx.draw_networkx_edges(G, pos, ax=ax, edgelist=edges, edge_color=edge_colors, edge_cmap=plt.cm.viridis, width=2)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=10, font_weight="bold")

    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=min_sim, vmax=max_sim))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("Similarity Score")

    # Save the graph in the output directory
    graph_path = output_dir / "similarity_graph.png"
    plt.savefig(graph_path, bbox_inches="tight")
    plt.close()

    print(f"Similarity graph saved at {graph_path}")

def generate_output_files(similarity_matrix, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    for filename, code in codes.items():
        ast_file = visualize_ast(code, output_dir / f"{filename}_ast.png")
        token_file = visualize_tokens(code, output_dir / f"{filename}_tokens.png")
        cfg_file = visualize_control_flow(code, output_dir / f"{filename}_cfg.png")

        print(f"Generated visualizations for {filename}:")
        print(f"  AST: {ast_file}")
        print(f"  Tokens: {token_file}")
        print(f"  Control Flow: {cfg_file}")
    

def main():
    # Define your input folder and output directory
    folder_path = r"C:\Users\daksh\Desktop\plag_check\buy_and_sell_stock"  # Your source code folder
    output_dir = Path("output_files")  # Output directory for visualizations
    output_dir.mkdir(exist_ok=True)

    # Load all code files
    codes = load_codes_from_folder(folder_path)

    # Compute pairwise similarities
    similarities = calculate_pairwise_similarities(codes)

    # Cluster based on similarity
    clusters, similarity_matrix, files = cluster_codes(similarities, codes)

    # Print clustering result
    print("\n🔍 Code Clusters based on AST/Token/Control Flow Similarity:")
    for cluster_id, cluster_files in clusters.items():
        if cluster_id == -1:
            print("🟡 Outliers:", cluster_files)
        else:
            print(f"🟢 Cluster {cluster_id}: {cluster_files}")

    #  Visualize Similarity Heatmap and Graph
    visualize_similarity_matrix(similarity_matrix, files, output_dir)
    visualize_similarity_graph(similarity_matrix, files, output_dir)

    #  Generate per-code AST, Token, CFG visualizations
    generate_output_files(output_dir, codes)

    print(f"\n✅ All outputs saved in: {output_dir.resolve()}")


# Make sure generate_output_files takes codes
def generate_output_files( output_dir, codes):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    for filename, code in codes.items():
        base_name = Path(filename).stem  # Strip ".py" if present
        ast_file = visualize_ast(code, output_dir / f"{base_name}_ast.png")
        token_file = visualize_tokens(code, output_dir / f"{base_name}_tokens.png")
        cfg_file = visualize_control_flow(code, output_dir / f"{base_name}_cfg.png")

        print(f"✅ Visualizations for {filename}:")
        print(f"  AST: {ast_file}")
        print(f"  Tokens: {token_file}")
        print(f"  Control Flow: {cfg_file}")


if __name__ == "__main__":
    main()

