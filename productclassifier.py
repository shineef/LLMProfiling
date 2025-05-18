import pandas as pd
import transformers
import torch
from huggingface_hub import login
import json

import matplotlib.pyplot as plt

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
            {"role": "system", "content": "Based on the given product description, determine the most appropriate category. Output only the corresponding category number (no additional text or formatting). Part of the 47 categories are: 0,Home & Kitchen; 1,Health & Personal Care; 2,Beauty; 3,Sports & Outdoors; 4,Books; 5,'Patio, Lawn & Garden'; 6,Toys & Games; 7,CDs & Vinyl; 8,Cell Phones & Accessories; 9,Grocery & Gourmet Food; 10,'Arts, Crafts & Sewing'; 11,'Clothing, Shoes & Jewelry'; 12,Electronics; 13,Movies & TV; 14,Software; 15,Video Games; 16,Automotive; 17,Pet Supplies; 18,Office Products; 19,Industrial & Scientific; 20,Musical Instruments; 21,Tools & Home Improvement; 22,Magazine Subscriptions; 23,Baby Products; 25,Appliances; 26,Kitchen & Dining; 27,Collectibles & Fine Art; 28,All Beauty; 29,Luxury Beauty; 30,Amazon Fashion; 31,Computers; 32,All Electronics; 33,Purchase Circles; 34,MP3 Players & Accessories; 35,Gift Cards; 36,Office & School Supplies; 37,Home Improvement; 38,Camera & Photo; 39,GPS & Navigation; 40,Digital Music; 41,Car Electronics; 42,Baby; 43,Kindle Store; 44,Buy a Kindle; 45,Furniture & Decor"},        
            {"role": "user", "content": f"{description}"},
        ]

        outputs = pipeline(
        messages,
        max_new_tokens=128000,
    )
        return outputs[0]["generated_text"][-1]['content']

    descriptions = pd.read_csv('idtest_data.tsv', delimiter='\t', header=None, names=['node_idx', 'description'])

    edges = pd.read_csv('ogbn-products/raw/edge.csv.gz', header=None, names=['source', 'target'], compression='gzip')

    description_p = dict(zip(descriptions['node_idx'], descriptions['description']))

    node_descriptions = {}

    for idx in descriptions['node_idx']:
        node_descriptions[idx] = f"'{description_p[idx]}"

    neighbors = {}

    for _, edge in edges.iterrows():
        neighbors.setdefault(edge['source'], []).append(edge['target'])
        neighbors.setdefault(edge['target'], []).append(edge['source'])

    acc_results = []

    neighbor_limits = [0, 5, 10]

    for limit in neighbor_limits:
        for node, description in node_descriptions.items():
            if node in neighbors:
                count = 0
                for neighbor in neighbors[node]:
                    if int(neighbor) not in description_p:
                        continue
                    if count < limit:
                        # neighbor_title = titles[neighbor]
                        # neighbor_abstract = abstracts[neighbor]
                        neighbor_discription = description_p[neighbor]
                        # description += f" linked with node: {neighbor_title}."
                        description += f" - linked with node '{neighbor_discription}'.\n"
                        count += 1
                    else:
                        break
                node_descriptions[node] = description

        with open(f'node_descriptions_products_{limit}_test_origin.json', 'w') as file:
            json.dump(node_descriptions, file)

        with open(f'node_descriptions_products_{limit}_test_origin.json', 'r') as file:
            loaded_descriptions = json.load(file)

        predictions = []

        for node, description in loaded_descriptions.items():
            predict_label = LLMClassifier(description)
            predictions.append(predict_label)

        predictions_df = pd.DataFrame({
            'Node': list(loaded_descriptions.keys()),
            'PredictedLabel': predictions
        })

        predictions_df.to_csv(f'predictions_{limit}_test_ProductsLlamaPro_sub{seed}.csv', index=False)

        predictions_df = pd.read_csv(f'predictions_{limit}_test_ProductsLlamaPro_sub{seed}.csv')

        predictions_df['PredictedLabel'] = pd.to_numeric(predictions_df['PredictedLabel'], errors='coerce')

        correct_labels = pd.read_csv(f'test0_labels_seed_{seed}.csv', header=None, names=['TrueLabel', 'Node'])
        correct_labels['TrueLabel'] = correct_labels['TrueLabel'].astype(float)

        aligned_labels = correct_labels.set_index('Node').loc[predictions_df['Node']].reset_index()

        accuracy = (predictions_df['PredictedLabel'].values == aligned_labels['TrueLabel'].values).mean()
        print(f"Accuracy: {accuracy * 100:.2f}%")


        all_test_results.append(accuracy)


mean_performance = torch.mean(torch.tensor(all_test_results))
std_performance = torch.std(torch.tensor(all_test_results), unbiased=True)

print(f"Mean Test Performance: {mean_performance:.4f}")
print(f"Unbiased Std Dev: {std_performance:.4f}")