import pandas as pd
import transformers
import torch
from huggingface_hub import login
import json

import matplotlib.pyplot as plt

from collections import deque

torch.set_default_tensor_type(torch.cuda.HalfTensor)


import random
import numpy as np

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior
    torch.backends.cudnn.benchmark = False

all_test_results = []

for seed in range(10):
    print(f"Running experiment with seed {seed}...")
    set_seed(seed)


    login(token="")

    model_id = ""

    pipeline = transformers.pipeline(
        "text-generation",
        model = model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        # device_map="auto",
        device=0,
    )

    def LLMClassifier(description):

        messages = [
            {"role": "system", "content": "You are a classifier that determines which category a given node belongs to from arxiv cs. The 40 categories are: 0-(Numerical Analysis) 1-(Multimedia) 2-(Logic in Computer Science) 3-(Computers and Society) 4-(Cryptography and Security) 5-(Distributed, Parallel, and Cluster Computing) 6-(Human-Computer Interaction) 7-(Computational Engineering, Finance, and Science) 8-(Networking and Internet Architecture) 9-(Computational Complexity)\
            10-(Artificial Intelligence) 11-(Multiagent Systems) 12-(General Literature) 13-(Neural and Evolutionary Computing) 14-(Symbolic Computation) 15-(Hardware Architecture) 16-(Computer Vision and Pattern Recognition) 17-(Graphics) 18-(Emerging Technologies) 19-(Systems and Control)\
            20-(Computational Geometry) 21-(Other Computer Science) 22-(Programming Languages) 23-(Software Engineering) 24-(Machine Learning) 25-(Sound) 26-(Social and Information Networks) 27-(Robotics) 28-(Information Theory) 29-(Performance)\
            30-(Computation and Language) 31-(Information Retrieval) 32-(Mathematical Software) 33-(Formal Languages and Automata Theory) 3 4-(Data Structures and Algorithms) 35-(Operating Systems) 36-(Computer Science and Game Theory) 37-(Databases) 38-(Digital Libraries) 39-(Discrete Mathematics). Output the corresponding number only and nothing else."},
            {"role": "user", "content": f"{description}"},
        ]

        outputs = pipeline(
        messages,
        max_new_tokens=128000,
    )
        return outputs[0]["generated_text"][-1]['content']

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
                # print(node_descriptions[node])
            
            for neighbor in neighbors.get(curr_node, []):
                queue.append((neighbor, hop + 1))

    titleabs = pd.read_csv('valid_dataset_origin.tsv', delimiter='\t')

    edges = pd.read_csv('ogbn_arxiv/raw/edge.csv.gz', header=None, names=['source', 'target'], compression='gzip')

    titles = dict(zip(titleabs['node_idx'], titleabs['title']))
    abstracts = dict(zip(titleabs['node_idx'], titleabs['abstract']))

    node_descriptions = {}

    for idx in titleabs['node_idx']:
        node_descriptions[idx] = f"Node Title: '{titles[idx]}'\nAbstract: '{abstracts[idx]}'\n\nNeighbors:\n"

    neighbors = {}

    for _, edge in edges.iterrows():
        neighbors.setdefault(edge['source'], []).append(edge['target'])
        neighbors.setdefault(edge['target'], []).append(edge['source'])

    acc_results = []
    neighbor_limits = [0, 5, 10]

    for limit in neighbor_limits:
        max_hops = 1
        for node in node_descriptions:
            bfs(node, max_hops, limit)

        with open(f'node_descriptions_abs_{limit}_test_origin.json', 'w') as file:
            json.dump(node_descriptions, file)

        with open(f'node_descriptions_abs_{limit}_test_ArxivLlama.json', 'r') as file:
            loaded_descriptions = json.load(file)

        predictions = []

        for node, description in loaded_descriptions.items():
            predict_label = LLMClassifier(description)
            predictions.append(predict_label)

        predictions_df = pd.DataFrame(predictions, columns=['PredictedLabel'])

        predictions_df.to_csv(f'predictions_abs_{limit}_valid_ArxivLlama.csv', index=False)

        predictions_df = pd.read_csv(f'predictions_abs_{limit}_valid_ArxivLlama.csv')
        predictions_df['PredictedLabel'] = pd.to_numeric(predictions_df['PredictedLabel'], errors='coerce')

        correct_labels = pd.read_csv('valid_label.csv.gz', header=None, names=['TrueLabel'])
        correct_labels['TrueLabel'] = correct_labels['TrueLabel'].astype(float)

        accuracy = (predictions_df['PredictedLabel'] == correct_labels['TrueLabel']).mean()
        print(f"Accuracy: {accuracy * 100:.2f}%")


        all_test_results.append(accuracy)


mean_performance = torch.mean(torch.tensor(all_test_results))
std_performance = torch.std(torch.tensor(all_test_results), unbiased=True)

print(f"Mean Test Performance: {mean_performance:.4f}")
print(f"Unbiased Std Dev: {std_performance:.4f}")
