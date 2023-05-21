import time
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import numpy as np
import warnings
import pandas as pd
import rdkit, rdkit.Chem, rdkit.Chem.rdDepictor, rdkit.Chem.Draw
import networkx as nx
import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import torch.utils.data as data
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import dense_to_sparse, add_self_loops, to_scipy_sparse_matrix
from torch_geometric.data import Data
import torch.nn.functional as F
from six.moves import urllib
import deepchem as dc
import random
from torch_geometric.nn.aggr import Set2Set
import multiprocessing
from torchvision.transforms import Compose, ToTensor

def get_data():
    # had to rehost because dataverse isn't reliable
    soldata = pd.read_csv(
        # "https://github.com/whitead/dmol-book/raw/main/data/curated-solubility-dataset.csv"
        "/Users/adityabehal/Downloads/curated-solubility-dataset.csv"
    )

    return soldata


def gen_smiles2graph(smiles):
    """Argument for the RD2NX function should be a valid SMILES sequence
    returns: the graph
    """
    featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=True)
    out = featurizer.featurize(smiles)
    return out[0]


def featurize_data():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    soldata = get_data()

    graph = []
    sol = []

    # currentNumInstances = 0
    # maxInstances = 100

    # for i in range(100):
    for i in range(len(soldata)):
        # if currentNumInstances == maxInstances:
        # break
        graphInstance = gen_smiles2graph(soldata.SMILES[i])
        if hasattr(graphInstance, "node_features") and hasattr(graphInstance, "edge_index") and hasattr(graphInstance,
                                                                                                        "edge_features"):
            graph.append(graphInstance)
            sol.append(soldata.Solubility[i])
            # currentNumInstances += 1

    return graph, sol


class CustomDataset(data.Dataset):
    def __init__(self, graphAll, solAll, transform=None, target_transform=None):
        self.graphInstances = graphAll
        self.solInstances = solAll
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.graphInstances)

    def __getitem__(self, idx):
        graphInstance = self.graphInstances[idx]
        solInstance = self.solInstances[idx]

        if self.transform:
            graphInstanceNodeFeatures = self.transform(graphInstance.node_features)
            graphInstanceEdgeIndex = self.transform(graphInstance.edge_index)
            graphInstanceEdgeFeatures = self.transform(graphInstance.edge_features)

            # B = torch.reshape(A, (A.shape[1], A.shape[2]))
            graphInstanceNodeFeatures = torch.reshape(graphInstanceNodeFeatures, (
            graphInstanceNodeFeatures.shape[1], graphInstanceNodeFeatures.shape[2]))
            graphInstanceEdgeIndex = torch.reshape(graphInstanceEdgeIndex,
                                                   (graphInstanceEdgeIndex.shape[1], graphInstanceEdgeIndex.shape[2]))
            graphInstanceEdgeFeatures = torch.reshape(graphInstanceEdgeFeatures, (
            graphInstanceEdgeFeatures.shape[1], graphInstanceEdgeFeatures.shape[2]))

        if self.target_transform:
            solInstance = self.target_transform(solInstance)

        return graphInstanceNodeFeatures, graphInstanceEdgeIndex, graphInstanceEdgeFeatures, solInstance


def train_val_test_split():
    graph, sol = featurize_data()

    dataset = CustomDataset(graphAll=graph, solAll=sol, transform=Compose([ToTensor()]))

    cores = multiprocessing.cpu_count()  # Count the number of cores in a computer (includes hyperthreading)
    print("cores: ", cores)

    numWorkersToUse = int(cores/2)  # 16 cores (with hyperthreading) on my machine - using 8 has worked well in the past

    # batch_size=1 was original default - 100 was used in bondnet paper
    # set num_workers=cores for best performance
    dataloader = data.DataLoader(dataset, batch_size=1,
                                 shuffle=True, num_workers=numWorkersToUse)
    # print(len(dataloader))
    # print(type(dataloader))

    test_data_size = int(0.1 * len(dataloader))
    val_data_size = int(0.1 * len(dataloader))
    train_data_size = len(dataloader) - 2 * int(0.1 * len(dataloader))

    # test_data_size = 20
    # val_data_size = 20
    # train_data_size = 160

    # test_data_size = 200
    # val_data_size = 200
    # train_data_size = len(dataloader) - 400

    print(test_data_size)
    print(val_data_size)
    print(train_data_size)

    test_data, val_data, train_data = data.random_split(dataloader, [test_data_size, val_data_size, train_data_size])

    return train_data

train_data = train_val_test_split()

pin_memory = True
print('pin_memory is', pin_memory)

for num_workers in range(0, 17, 1):
    train_loader = data.DataLoader(train_data, batch_size=1, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)

    start = time.time()
    for epoch in range(1, 5):
        for batch, (X1, X2, X3, y) in enumerate(train_loader.dataset.dataset):
            pass
    end = time.time()
    print("Finish with:{} second, num_workers={}".format(end - start, num_workers))