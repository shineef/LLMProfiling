import pandas as pd

nodeidx2paperid = pd.read_csv('ogbn_arxiv/mapping/nodeidx2paperid.csv.gz', compression='gzip')

titleabs = pd.read_csv('titleabs.tsv', sep='\t', header=None, names=['paper_id', 'title', 'abstract'])

nodeidx2paperid['paper id'] = nodeidx2paperid['paper id'].astype(int)

filtered_titleabs = titleabs[titleabs['paper_id'].isin(nodeidx2paperid['paper id'])]

merged_data = pd.merge(filtered_titleabs, nodeidx2paperid, left_on='paper_id', right_on='paper id', how='left')

final_data = merged_data.drop(columns=['paper_id', 'paper id']).rename(columns={'node idx': 'node_idx'})

final_data_sorted = final_data.sort_values(by='node_idx')

final_data_sorted.reset_index(drop=True, inplace=True)
final_data_sorted = final_data_sorted[['node_idx', 'title', 'abstract']]
print(final_data_sorted.head())

final_data_sorted.to_csv('filtered_titleabs_origin.tsv', sep='\t', index=False)