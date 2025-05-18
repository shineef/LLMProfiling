import pandas as pd
import transformers
import torch
from huggingface_hub import login
import json

import matplotlib.pyplot as plt

torch.set_default_tensor_type(torch.cuda.HalfTensor)

login(token="")

model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

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
    max_new_tokens=131072,
)
    return outputs[0]["generated_text"][-1]['content']

titleabs = pd.read_csv('filtered_titleabs.tsv', delimiter='\t')

edges = pd.read_csv('ogbn_arxiv/raw/edge.csv.gz', header=None, names=['source', 'target'], compression='gzip')

titles = dict(zip(titleabs['node_idx'], titleabs['title']))
abstracts = dict(zip(titleabs['node_idx'], titleabs['abstract']))

node_descriptions = {}

for idx in titleabs['node_idx']:
    node_descriptions[idx] = f"node has the title: '{titles[idx]}' and abstract: '{abstracts[idx]}.'"

neighbors = {}

for _, edge in edges.iterrows():
    neighbors.setdefault(edge['source'], []).append(edge['target'])
    neighbors.setdefault(edge['target'], []).append(edge['source'])

acc_results = []
neighbor_limits = [10, 15, 20, 25]

for limit in neighbor_limits:
    for node, description in node_descriptions.items():
        if node in neighbors:
            count = 0
            for neighbor in neighbors[node]:
                if count < limit:
                    neighbor_title = titles[neighbor]
                    # neighbor_abstract = abstracts[neighbor]
                    description += f" linked with node: {neighbor_title}."
                    # description += f" linked with node {neighbor_title} which abstract is: {neighbor_abstract}."
                    count += 1
                else:
                    break
            node_descriptions[node] = description

    with open(f'node_descriptions_abs_{limit}_0shot.json', 'w') as file:
        json.dump(node_descriptions, file)

    with open(f'node_descriptions_abs_{limit}_0shot.json', 'r') as file:
        loaded_descriptions = json.load(file)

    predictions = []

    for node, description in loaded_descriptions.items():
        # print(node)
        # print(description)
        predict_label = LLMClassifier(description)
        predictions.append(predict_label)

    predictions_df = pd.DataFrame(predictions, columns=['PredictedLabel'])

    predictions_df.to_csv(f'predictions_abs_emp_{limit}_0shot.csv', index=False)

    predictions_df = pd.read_csv(f'predictions_abs_emp_{limit}_0shot.csv')
    predictions_df['PredictedLabel'] = pd.to_numeric(predictions_df['PredictedLabel'], errors='coerce')

    print(predictions_df.iloc[0])

    correct_labels = pd.read_csv('ogbn_arxiv/raw/node-label.csv.gz', header=None, names=['TrueLabel'])
    correct_labels['TrueLabel'] = correct_labels['TrueLabel'].astype(float)

    print(correct_labels.iloc[0])

    accuracy = (predictions_df['PredictedLabel'] == correct_labels['TrueLabel']).mean()
    print(f"Accuracy: {accuracy * 100:.2f}%")

    acc_results.append(accuracy)