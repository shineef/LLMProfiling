import csv
import pandas as pd

import sys
csv.field_size_limit(sys.maxsize)

with open("ogbn-products/X.all.txt", "r", encoding="utf-8") as f:
    data_lines = f.readlines()

valid_indices = []
for i, item in enumerate(data_lines[235938:], start=235938):  
        valid_indices.append(i)

print(len(valid_indices))

input_file = "test_data.tsv"
output_file = "idtest_data.tsv"

with open(input_file, "r", encoding="utf-8") as infile, \
     open(output_file, "w", encoding="utf-8", newline="") as outfile:
    reader = csv.reader(infile, delimiter="\t")
    writer = csv.writer(outfile, delimiter="\t")
    
    for idx, row in zip(valid_indices, reader):
        writer.writerow([idx] + row) 

df = pd.read_csv(output_file, delimiter='\t', header=None, names=['node_idx', 'description'])
descriptions = dict(zip(df['node_idx'], df['description']))
