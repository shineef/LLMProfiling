from ogb.graphproppred import PygGraphPropPredDataset

dataset1 = PygGraphPropPredDataset(name = "ogbn-arxiv", root = 'dataset/')
dataset2 = PygGraphPropPredDataset(name = "ogbn-products", root = 'dataset/')