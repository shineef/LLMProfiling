import pandas as pd
import json
import gzip

from collections import deque
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr


titleabs = pd.read_csv('train_bart.tsv', delimiter='\t')

edges = pd.read_csv('ogbn_arxiv/raw/edge.csv.gz', header=None, names=['source', 'target'], compression='gzip')

titles = dict(zip(titleabs['node_idx'], titleabs['title']))
abstracts = dict(zip(titleabs['node_idx'], titleabs['abstract']))

def calculate_similarity_scores(node1, node2):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([abstracts[node1], abstracts[node2]])
    cosine_sim = cosine_similarity(tfidf_matrix)[0][1]

    set1 = set(abstracts[node1].split())
    set2 = set(abstracts[node2].split())
    jaccard_sim = len(set1.intersection(set2)) / len(set1.union(set2))

    return cosine_sim, jaccard_sim

node_descriptions = {}

for idx in titleabs['node_idx']:
    node_descriptions[idx] = f"Node Title: '{titles[idx]}'\nAbstract: '{abstracts[idx]}'\n\nNeighbors:\n"

neighbors = {}

for _, edge in edges.iterrows():
    neighbors.setdefault(edge['source'], []).append(edge['target'])
    neighbors.setdefault(edge['target'], []).append(edge['source'])

def bfs(node, max_hops, max_neighbors):
    visited = set()
    queue = deque([(node, 0)])
    neighbor_count = 0
    
    while queue and neighbor_count < max_neighbors:
        curr_node, hop = queue.popleft()
        if curr_node in visited or hop > max_hops:
            continue
        
        visited.add(curr_node)
        
        if curr_node != node and curr_node in titles:
            # cosine_sim, jaccard_sim = calculate_similarity_scores(node, curr_node)
            neighbor_title = titles[curr_node]
            neighbor_abstract = abstracts[curr_node]
            # node_descriptions[node] += f"- Node Title: '{neighbor_title}' ({hop} hop)\n  Abstract: '{neighbor_abstract}'\n  Similarity Scores:\n    Cosine Similarity: {cosine_sim:.3f}\n    Jaccard Similarity: {jaccard_sim:.3f}\n"
            node_descriptions[node] += f"- Node Title: '{neighbor_title}' ({hop} hop)\n  Abstract: '{neighbor_abstract}'\n"
            neighbor_count += 1
        
        for neighbor in neighbors.get(curr_node, []):
            queue.append((neighbor, hop + 1))

max_hops = 2 
max_neighbors = 5

for node in node_descriptions:
    bfs(node, max_hops, max_neighbors)

with open(f'node_descriptions_abs_2h5n_train_bart.json', 'w') as file:
    json.dump(node_descriptions, file)


with gzip.open('ogbn_arxiv/raw/node-label.csv.gz', 'rt') as f:
    labels = f.read().splitlines()

with open(f'node_descriptions_abs_2h5n_train_bart.json', 'r') as file:
        loaded_descriptions = json.load(file)

json_data = []

for node, description in loaded_descriptions.items():
    label = labels[int(node)]

    conversation = [
        {"from": "system", "value": "You are a classifier that determines which category a given node belongs to from arxiv cs. The 40 categories are: 0-(Numerical Analysis) 1-(Multimedia) 2-(Logic in Computer Science) 3-(Computers and Society) 4-(Cryptography and Security) 5-(Distributed, Parallel, and Cluster Computing) 6-(Human-Computer Interaction) 7-(Computational Engineering, Finance, and Science) 8-(Networking and Internet Architecture) 9-(Computational Complexity)\
          10-(Artificial Intelligence) 11-(Multiagent Systems) 12-(General Literature) 13-(Neural and Evolutionary Computing) 14-(Symbolic Computation) 15-(Hardware Architecture) 16-(Computer Vision and Pattern Recognition) 17-(Graphics) 18-(Emerging Technologies) 19-(Systems and Control)\
          20-(Computational Geometry) 21-(Other Computer Science) 22-(Programming Languages) 23-(Software Engineering) 24-(Machine Learning) 25-(Sound) 26-(Social and Information Networks) 27-(Robotics) 28-(Information Theory) 29-(Performance)\
          30-(Computation and Language) 31-(Information Retrieval) 32-(Mathematical Software) 33-(Formal Languages and Automata Theory) 34-(Data Structures and Algorithms) 35-(Operating Systems) 36-(Computer Science and Game Theory) 37-(Databases) 38-(Digital Libraries) 39-(Discrete Mathematics). Output the corresponding number ONLY"},        
        {"from": "human", "value": f"{description}"},
        {"from": "gpt", "value": f"{label}"}
    ]

    json_data.append({"conversations": conversation})

with open('train_dataset_bart_2hop5n.json', 'w', encoding='utf-8') as f:
    json.dump(json_data, f, ensure_ascii=False, indent=4)
