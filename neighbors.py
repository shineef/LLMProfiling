import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

titleabs = pd.read_csv('/home/xfang1/dataset/ogbn_arxiv_td/valid_dataset.tsv', delimiter='\t')

edges = pd.read_csv('/home/xfang1/dataset/ogbn_arxiv/raw/edge.csv.gz', header=None, names=['source', 'target'], compression='gzip')

neighbors = {}
for _, edge in edges.iterrows():
    neighbors.setdefault(edge['source'], []).append(edge['target'])
    neighbors.setdefault(edge['target'], []).append(edge['source'])

neighbor_counts = {node: len(neighbors.get(node, [])) for node in titleabs['node_idx']}

bins = list(range(0, 101, 10))  # [0, 10, 20, ..., 100]
neighbor_distribution = Counter((count // 10) * 10 for count in neighbor_counts.values() if count <= 100)
neighbor_distribution = {bin: neighbor_distribution.get(bin, 0) for bin in bins}
neighbor_distribution[110] = sum(1 for count in neighbor_counts.values() if count > 100)  

neighbor_counts_list = sorted(neighbor_counts.values())
percentiles = [50, 60, 70, 75, 80, 85, 90, 95, 97, 98, 98.5, 99]
percentile_values = {p: np.percentile(neighbor_counts_list, p) for p in percentiles}

for bin, count in sorted(neighbor_distribution.items()):
    if bin <= 100:
        print(f"neighbor {bin}-{bin + 9} number: {count}")
    else:
        print(f"neighbor >100 number: {count}")

print("\npercentile values：")
for p, value in percentile_values.items():
    print(f"{p}% percentile: {value:.2f}")

max_neighbors = max(neighbor_counts.values())
total_count_in_bins = sum(neighbor_distribution.values())
total_nodes = len(titleabs['node_idx'])

print(f"\nmaximum number of neighbors: {max_neighbors}")
print(f"sum bins: {total_count_in_bins}")
print(f"sum nodes: {total_nodes}")

plt.figure(figsize=(10, 6))
x_labels = [f"{bin}-{bin+9}" if bin <= 100 else ">100" for bin in sorted(neighbor_distribution.keys())]
y_values = [neighbor_distribution[bin] for bin in sorted(neighbor_distribution.keys())]

plt.bar(x_labels, y_values, color='skyblue', edgecolor='black')
plt.title("Neighbor Number Distribution", fontsize=16)
plt.xlabel("Neighbor Number Range", fontsize=14)
plt.ylabel("Number of Nodes", fontsize=14)
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

plt.savefig('Arxiv Neighbor Number Distribution.png', format='png', dpi=300)
plt.close()

plt.show()