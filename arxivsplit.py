import pandas as pd
import gzip

original_dataset_path = "filtered_titleabs_origin.tsv"  
df = pd.read_csv(original_dataset_path, sep='\t')

def read_gzipped_csv_no_header(file_path):
    with gzip.open(file_path, 'rt') as f:
        return pd.read_csv(f, header=None, names=["node_idx"]) 

train_idx = read_gzipped_csv_no_header("ogbn_arxiv/split/time/train.csv.gz")["node_idx"]
valid_idx = read_gzipped_csv_no_header("ogbn_arxiv/split/time/valid.csv.gz")["node_idx"]
test_idx = read_gzipped_csv_no_header("ogbn_arxiv/split/time/test.csv.gz")["node_idx"]

train_count = len(train_idx)
valid_count = len(valid_idx)
test_count = len(test_idx)

total_count = train_count + valid_count + test_count
train_ratio = train_count / total_count
valid_ratio = valid_count / total_count
test_ratio = test_count / total_count

print(f"Train count: {train_count}, Ratio: {train_ratio:.2%}")
print(f"Valid count: {valid_count}, Ratio: {valid_ratio:.2%}")
print(f"Test count: {test_count}, Ratio: {test_ratio:.2%}")

assert train_idx.is_unique and valid_idx.is_unique and test_idx.is_unique

assert set(train_idx).isdisjoint(valid_idx)
assert set(train_idx).isdisjoint(test_idx)
assert set(valid_idx).isdisjoint(valid_idx)

train_set = df[df["node_idx"].isin(train_idx)]
valid_set = df[df["node_idx"].isin(valid_idx)]
test_set = df[df["node_idx"].isin(test_idx)]

train_set.to_csv("train_dataset_origin.tsv", sep='\t', index=False)
valid_set.to_csv("valid_dataset_origin.tsv", sep='\t', index=False)
test_set.to_csv("test_dataset_origin.tsv", sep='\t', index=False)

