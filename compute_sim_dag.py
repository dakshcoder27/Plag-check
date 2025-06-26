import os
import re
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 🔧 Utility: Clean and tokenize code
def preprocess_code(code):
    code = re.sub(r'//.*|/\*[\s\S]*?\*/|#.*', '', code)  # Remove comments
    code = re.sub(r'\s+', ' ', code)  # Normalize whitespace
    return code.strip()

# 📁 Load all code files
def load_code_files(folder_path):
    files = []
    contents = []
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith(".py") or filename.endswith(".cpp") or filename.endswith(".java"):
            with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as file:
                files.append(filename)
                contents.append(preprocess_code(file.read()))
    return files, contents

# 🧠 Compute cosine similarity and build DAG
def compute_similarity_dag(code_folder, threshold=0.6):
    files, code_snippets = load_code_files(code_folder)
    
    # Vectorize using TF-IDF
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(code_snippets)
    
    # Cosine similarity matrix
    similarity = cosine_similarity(tfidf_matrix)
    
    # 📊 Build DAG
    G = nx.DiGraph()
    G.add_nodes_from(files)

    for i in range(len(files)):
        for j in range(i+1, len(files)):  # Only i -> j to prevent cycles
            if similarity[i][j] > threshold:
                G.add_edge(files[i], files[j], weight=similarity[i][j])
    
    return G, similarity, files

# 🔍 Example usage
folder = r"C:\Users\daksh\Desktop\plag_check\buy_and_sell_stock"
dag, sim_matrix, filenames = compute_similarity_dag(folder)

# 📈 Visualize
import matplotlib.pyplot as plt

pos = nx.spring_layout(dag)
nx.draw(dag, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=2000)
edge_labels = nx.get_edge_attributes(dag, 'weight')
nx.draw_networkx_edge_labels(dag, pos, edge_labels={k: f"{v:.2f}" for k, v in edge_labels.items()})
plt.title("Code Similarity DAG")
plt.show()
