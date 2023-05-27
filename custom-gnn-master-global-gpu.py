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
from sklearn.preprocessing import MinMaxScaler

def get_data():
    # had to rehost because dataverse isn't reliable
    soldata = pd.read_csv(
        "https://github.com/whitead/dmol-book/raw/main/data/curated-solubility-dataset.csv"
        # "/Users/adityabehal/Downloads/curated-solubility-dataset.csv"
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

    all_global_features = np.zeros((len(soldata), 6))

    kept_graph_indices = []

    # currentNumInstances = 0
    # maxInstances = 100

    global_feature_counter = 0

    # for i in range(100):
    for i in range(len(soldata)):
        # if currentNumInstances == maxInstances:
            # break
        graphInstance = gen_smiles2graph(soldata.SMILES[i])
        if hasattr(graphInstance, "node_features") and hasattr(graphInstance, "edge_index") and hasattr(graphInstance, "edge_features"):
            kept_graph_indices.append(i)

            # number of nodes, edges (technically 2x the number of actual edges), molecular weight,
            # and one hot encoding of molecular formal charge per table S2 in bondnet supplemental information
            all_global_features[global_feature_counter][0] = graphInstance.num_nodes
            all_global_features[global_feature_counter][1] = graphInstance.num_edges
            all_global_features[global_feature_counter][2] = round(rdkit.Chem.Descriptors.ExactMolWt(rdkit.Chem.MolFromSmiles("CO")))

            formal_charge = rdkit.Chem.rdmolops.GetFormalCharge(rdkit.Chem.MolFromSmiles("CO"))

            if formal_charge < 0:
                all_global_features[global_feature_counter][3] = 1
                all_global_features[global_feature_counter][4] = 0
                all_global_features[global_feature_counter][5] = 0
            elif formal_charge > 0:
                all_global_features[global_feature_counter][3] = 0
                all_global_features[global_feature_counter][4] = 0
                all_global_features[global_feature_counter][5] = 1
            else:
                all_global_features[global_feature_counter][3] = 0
                all_global_features[global_feature_counter][4] = 1
                all_global_features[global_feature_counter][5] = 0

            global_feature_counter += 1

            # ask das about global features normalization and conversion of moleculr weight to integer (should we do this? if so, truncate or round?)
            # graphInstance.z = global_features
            '''
            graphInstanceWithGlobalFeatures = dc.feat.graph_data.GraphData(node_features=graphInstance.node_features,
                                             edge_index=graphInstance.edge_index,
                                             edge_features=graphInstance.edge_features,
                                             z=global_features)
            '''
            graph.append(graphInstance)
            sol.append(soldata.Solubility[i])

            # currentNumInstances += 1

    # remove rows of zeros from 2D array (these correspond to graphs that were never properly featurized
    # and will not be included in the dataset)
    all_global_features = all_global_features[~np.all(all_global_features == 0, axis=1)]

    scaler = MinMaxScaler()
    all_global_features[:, :3] = scaler.fit_transform(all_global_features[:, :3])

    for i in range(len(graph)):
        graph[i].z = all_global_features[i].reshape(1, 6)
        # print("graphInstance global features: ")
        # print(graph[i].z)

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
            graphInstanceGlobalFeatures = self.transform(graphInstance.z)

            # B = torch.reshape(A, (A.shape[1], A.shape[2]))
            graphInstanceNodeFeatures = torch.reshape(graphInstanceNodeFeatures, (
            graphInstanceNodeFeatures.shape[1], graphInstanceNodeFeatures.shape[2]))
            graphInstanceEdgeIndex = torch.reshape(graphInstanceEdgeIndex,
                                                   (graphInstanceEdgeIndex.shape[1], graphInstanceEdgeIndex.shape[2]))
            graphInstanceEdgeFeatures = torch.reshape(graphInstanceEdgeFeatures, (
            graphInstanceEdgeFeatures.shape[1], graphInstanceEdgeFeatures.shape[2]))
            graphInstanceGlobalFeatures = torch.reshape(graphInstanceGlobalFeatures, (
            graphInstanceGlobalFeatures.shape[1], graphInstanceGlobalFeatures.shape[2]))

        if self.target_transform:
            solInstance = self.target_transform(solInstance)

        return graphInstanceNodeFeatures, graphInstanceEdgeIndex, graphInstanceEdgeFeatures, graphInstanceGlobalFeatures, solInstance


def train_val_test_split():
    graph, sol = featurize_data()

    dataset = CustomDataset(graphAll=graph, solAll=sol, transform=Compose([ToTensor()]))

    cores = multiprocessing.cpu_count()  # Count the number of cores in a computer (includes hyperthreading)
    print("cores: ", cores)

    numWorkersToUse = 0 # because batch size is 1 right now

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

    # set num_workers=cores for best performance
    test_data_loader = data.DataLoader(test_data, batch_size=1, shuffle=True, num_workers=numWorkersToUse)
    val_data_loader = data.DataLoader(val_data, batch_size=1, shuffle=True, num_workers=numWorkersToUse)
    train_data_loader = data.DataLoader(train_data, batch_size=1, shuffle=True, num_workers=numWorkersToUse)
    # print(test_data)
    # print(val_data)
    # print(train_data)
    # print(train_data.__getitem__(0))

    return test_data_loader, val_data_loader, train_data_loader


# create "zeroth" FCNN with 1 fully connected layer
# condense node features from 30 to 24
# dilate edge features from 11 to 24
# see Table S4 BDNCM input feature embedding size 24: https://www.rsc.org/suppdata/d0/sc/d0sc05251e/d0sc05251e1.pdf
class InitialEmbedding(torch.nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.fc_initial_embedding = nn.Linear(c_in, c_out)

    def forward(self, features):
        features = self.fc_initial_embedding(features)
        features = F.relu(features)

        return features


# neural network with two fully connected layers
class FCNN(torch.nn.Module):
    # c_in1 = 24, c_out1 = 256, c_out2 = 24
    def __init__(self, c_in1, c_out1, c_out2):
        super().__init__()
        self.fc1 = nn.Linear(c_in1, c_out1)
        self.fc2 = nn.Linear(c_out1, c_out2)

    def forward(self, features):
        # print("input 2 layer FCNN features: ", features)
        features = self.fc1(features)
        # print("after 1 linear layer updated features: ", features)
        features = F.relu(features)
        # print("after ReLu updated features: ", features)
        features = self.fc2(features)

        # print("after 2 linear layers updated features: ", features)

        return features

# implementation of equation 4 in bondnet paper
# https://pubs.rsc.org/en/content/articlepdf/2021/sc/d0sc05251e
class EdgeFeatures(torch.nn.Module):
    # c_in1 = 24, c_out1 = 24, c_out2 = 24
    def __init__(self, c_in1, c_out1, c_out2):
        super().__init__()
        # self.fc_initial_embedding = InitialEmbedding(c_in=11, c_out=24)
        self.FCNN_one = FCNN(c_in1=c_in1, c_out1=c_out1, c_out2=c_out2)
        self.FCNN_two = FCNN(c_in1=c_in1, c_out1=c_out1, c_out2=c_out2)
        self.FCNN_three = FCNN(c_in1=c_in1, c_out1=c_out1, c_out2=c_out2)

    def forward(self, node_features, edge_index, edge_features, global_features):
        original_edge_features = edge_features.detach().clone()

        for i in range(edge_index.shape[1]):
            # summing node features involved in the given edge and transforming them
            firstNodeIndex = int(edge_index[0][0][i])
            secondNodeIndex = int(edge_index[0][1][i])
            node_features_sum = node_features[0][firstNodeIndex] + node_features[0][secondNodeIndex]
            intermediate_node_features = self.FCNN_one(node_features_sum.T)

            # transforming the features of the given edge
            intermediate_edge_feature = self.FCNN_two(edge_features[0][i].T)

            intermediate_global_features = self.FCNN_three(global_features[0][0].T)

            # merging node features with features of the given edge

            # UPDATE TO INCLUDE THIS AT SOME POINT
            # intermediate_node_features + intermediate_edge_feature --> batch normalization --> drop out --> then ReLu

            intermediate_features_relu_input = intermediate_node_features + intermediate_edge_feature + intermediate_global_features

            instanceNorm1dLayer = nn.InstanceNorm1d(intermediate_features_relu_input.size(dim=0))
            dropoutLayer = nn.Dropout(p=0.1)

            intermediate_features_relu_input = instanceNorm1dLayer(
                torch.reshape(intermediate_features_relu_input, (1, intermediate_features_relu_input.size(dim=0))))
            intermediate_features_relu_input = torch.reshape(intermediate_features_relu_input, (-1,))
            intermediate_features_relu_input = dropoutLayer(intermediate_features_relu_input)

            intermediate_features = F.relu(intermediate_features_relu_input)

            # updating edge features
            edge_features[0][i] = ((original_edge_features[0][i].T + intermediate_features).T).detach().clone()

        return edge_features

# implementation of equation 5 in bondnet paper
# https://pubs.rsc.org/en/content/articlepdf/2021/sc/d0sc05251e
class NodeFeatures(torch.nn.Module):
    # c_in1 = 24, c_out1 = 24, c_out2 = 24
    def __init__(self, c_in1, c_out1, c_out2):
        super().__init__()
        # self.fc_initial_embedding = InitialEmbedding(c_in=30, c_out=24)
        self.FCNN_one = FCNN(c_in1=c_in1, c_out1=c_out1, c_out2=c_out2)
        self.FCNN_two = FCNN(c_in1=c_in1, c_out1=c_out1, c_out2=c_out2)
        self.FCNN_three = FCNN(c_in1=c_in1, c_out1=c_out1, c_out2=c_out2)

    def forward(self, node_features, edge_index, edge_features, global_features):
        sigmoidFunction = torch.nn.Sigmoid()

        original_node_features = node_features.detach().clone()

        epsilon = 1e-7

        for i in range(node_features.shape[1]):
            # DOUBLE CHECK WITH DAS
            # intermediate_node_feature = self.FCNN_one(node_features[i].T)
            intermediate_node_feature = self.FCNN_one(original_node_features[0][i])

            other_nodes_indices = []
            other_edges_indices = []

            other_edges_numerators = []
            other_edges_denominator = epsilon

            '''
            print("node_features[i].T: ", node_features[i].T)
            print("node_features[i].T.size: ", node_features[i].T.size())

            print("intermediate_node_feature: ", intermediate_node_feature)
            print("intermediate_node_feature.size: ", intermediate_node_feature.size())
            '''

            for j in range(edge_index.shape[1]):
                if edge_index[0][0][j] == i:
                    other_nodes_indices.append(int(edge_index[0][1][j]))
                    other_edges_indices.append(j)
                if edge_index[0][1][j] == i:
                    other_nodes_indices.append(int(edge_index[0][0][j]))
                    other_edges_indices.append(j)

            for other_edge_index in other_edges_indices:
                # print("SIGMOID ALERT TEST TEST TEST: ", sigmoidFunction(edge_features[other_edge_index]))
                other_edges_numerators.append(sigmoidFunction(edge_features[0][other_edge_index]))
                other_edges_denominator += sigmoidFunction(edge_features[0][other_edge_index])

            # print("other_edges_numerators: ", other_edges_numerators)
            # print("other_edges_denominator: ", other_edges_denominator)

            for other_edge_numerator, other_node_index in zip(other_edges_numerators, other_nodes_indices):
                edge_hat = other_edge_numerator / other_edges_denominator
                # DOUBLE CHECK WITH DAS
                # other_node_updated = self.FCNN_two(node_features[other_node_index].T)
                other_node_updated = self.FCNN_two(original_node_features[0][other_node_index].T)
                intermediate_node_feature += edge_hat * other_node_updated

                # print("edge_hat: ", edge_hat)

            # print("intermediate_node_feature: ", intermediate_node_feature)
            # print("intermediate_node_feature.size: ", intermediate_node_feature.size())

            intermediate_node_feature += self.FCNN_three(global_features[0][0].T)

            # UPDATE TO INCLUDE THIS AT SOME POINT
            # intermediate_node_feature --> batch normalization --> drop out --> then ReLu
            # should I use batch norm 1D and what should my feature size be at this point?

            instanceNorm1dLayer = nn.InstanceNorm1d(intermediate_node_feature.size(dim=0))
            dropoutLayer = nn.Dropout(p=0.1)

            intermediate_node_feature = instanceNorm1dLayer(
                torch.reshape(intermediate_node_feature, (1, intermediate_node_feature.size(dim=0))))
            intermediate_node_feature = torch.reshape(intermediate_node_feature, (-1,))
            intermediate_node_feature = dropoutLayer(intermediate_node_feature)

            # node_features[i] = F.relu(intermediate_node_feature).T
            node_features[0][i] = ((original_node_features[0][i].T + F.relu(intermediate_node_feature)).T).detach().clone()

            # print("actually updated node_features[i]: ", node_features[0][i])
            # print("actually updated node_features[i].size(): ", node_features[0][i].size())
            # print("********** NODE UPDATED SUCCESSFULLY ****************")

        return node_features

class GlobalFeatures(torch.nn.Module):
    # c_in1 = 24, c_out1 = 24, c_out2 = 24
    def __init__(self, c_in1, c_out1, c_out2):
        super().__init__()
        # self.fc_initial_embedding = InitialEmbedding(c_in=30, c_out=24)
        self.FCNN_one = FCNN(c_in1=c_in1, c_out1=c_out1, c_out2=c_out2)
        self.FCNN_two = FCNN(c_in1=c_in1, c_out1=c_out1, c_out2=c_out2)
        self.FCNN_three = FCNN(c_in1=c_in1, c_out1=c_out1, c_out2=c_out2)

    def forward(self, node_features, edge_index, edge_features, global_features):
        original_global_features = global_features.detach().clone()

        intermediate_global_features = self.FCNN_one((torch.sum(node_features[0], dim=0)/(node_features[0].shape[0])).T) + \
                                       self.FCNN_two((torch.sum(edge_features[0], dim=0)/(edge_features[0].shape[0])).T) + \
                                       self.FCNN_three(global_features[0][0].T)

        instanceNorm1dLayer = nn.InstanceNorm1d(intermediate_global_features.size(dim=0))
        dropoutLayer = nn.Dropout(p=0.1)

        intermediate_global_features = instanceNorm1dLayer(
            torch.reshape(intermediate_global_features, (1, intermediate_global_features.size(dim=0))))
        intermediate_global_features = torch.reshape(intermediate_global_features, (-1,))
        intermediate_global_features = dropoutLayer(intermediate_global_features)

        global_features[0][0] = ((original_global_features[0][0].T + F.relu(intermediate_global_features)).T).detach().clone()

        return global_features

class Graph2Graph(torch.nn.Module):
    def __init__(self, c_in1, c_out1, c_out2):
        super().__init__()
        self.EdgeFeaturesConvolution = EdgeFeatures(c_in1, c_out1, c_out2)
        self.NodeFeaturesConvolution = NodeFeatures(c_in1, c_out1, c_out2)
        self.GlobalFeaturesConvolution = GlobalFeatures(c_in1, c_out1, c_out2)

    def forward(self, node_features, edge_index, edge_features, global_features):
        # print("node_features_shape: ", node_features.shape)
        # print("edge_features_shape: ", edge_features.shape)
        # ask das about this ordering
        edge_features = self.EdgeFeaturesConvolution(node_features, edge_index, edge_features, global_features)
        node_features = self.NodeFeaturesConvolution(node_features, edge_index, edge_features, global_features)
        global_features = self.GlobalFeaturesConvolution(node_features, edge_index, edge_features, global_features)

        return node_features, edge_features, global_features


class Features_Set2Set():
    def __init__(self, initial_dim_out):
        self.node_s2s = Set2Set(in_channels=initial_dim_out, processing_steps=6, num_layers=3)
        self.edge_s2s = Set2Set(in_channels=initial_dim_out, processing_steps=6, num_layers=3)

    def transform_then_concat(self, node_features, edge_features, global_features):
        # RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation: [torch.FloatTensor [1, 24]], which is output 0 of AsStridedBackward0, is at version 8; expected version 7 instead. Hint: the backtrace further above shows the operation that failed to compute its gradient. The variable in question was changed in there or anywhere later. Good luck!
        # error above occurs if node and edge input features for Set2Set are not copied first

        node_features_input = node_features.detach().clone()
        edge_features_input = edge_features.detach().clone()
        global_features_input = global_features.detach().clone()

        node_features_input = torch.reshape(node_features_input,
                                            (node_features_input.shape[1], node_features_input.shape[2]))
        edge_features_input = torch.reshape(edge_features_input,
                                            (edge_features_input.shape[1], edge_features_input.shape[2]))

        node_features_transformed = self.node_s2s(node_features_input)
        edge_features_transformed = self.edge_s2s(edge_features_input)

        node_features_transformed = torch.reshape(node_features_transformed, (-1,))
        edge_features_transformed = torch.reshape(edge_features_transformed, (-1,))
        global_features_transformed = torch.reshape(global_features_input, (-1,))

        concatenated_features = torch.cat((node_features_transformed, edge_features_transformed, global_features_transformed))

        return concatenated_features


class Graph2Property(torch.nn.Module):
    # c_in1 = 24, c_out1 = 256, c_out2 = 128, c_out3 = 64, c_out4 = 1
    def __init__(self, c_in1, c_out1, c_out2, c_out3, c_out4):
        super().__init__()
        self.fc1 = nn.Linear(c_in1, c_out1)
        self.fc2 = nn.Linear(c_out1, c_out2)
        self.fc3 = nn.Linear(c_out2, c_out3)
        self.fc4 = nn.Linear(c_out3, c_out4)

    def forward(self, features):
        features = self.fc1(features)
        features = F.relu(features)
        features = self.fc2(features)
        features = F.relu(features)
        features = self.fc3(features)
        features = F.relu(features)
        features = self.fc4(features)

        return features


class GraphNeuralNetwork(torch.nn.Module):
    def __init__(self, nodes_initial_dim_in=30, edges_initial_dim_in=11, global_initial_dim_in=6,
                 initial_dim_out=24, g2g_input_dim=120, g2g_hidden_dim=256, g2p_dim_1=256, g2p_dim_2=128, g2p_dim_3=64):
        super(GraphNeuralNetwork, self).__init__()
        self.nodes_initial_embedding = InitialEmbedding(nodes_initial_dim_in, initial_dim_out)
        self.edges_initial_embedding = InitialEmbedding(edges_initial_dim_in, initial_dim_out)
        self.global_initial_embedding = InitialEmbedding(global_initial_dim_in, initial_dim_out)
        self.g2g_module = Graph2Graph(initial_dim_out, g2g_hidden_dim, initial_dim_out)
        self.features_set2set = Features_Set2Set(initial_dim_out)
        self.g2p_module = Graph2Property(g2g_input_dim, g2p_dim_1, g2p_dim_2, g2p_dim_3, 1)

    # g2g_num's default should be 4, can be set to 1 for memory debugging purposes
    def forward(self, X1, X2, X3, X4, g2g_num=0):
        node_features = X1
        edge_index = X2
        edge_features = X3
        global_features = X4

        node_features_updated = self.nodes_initial_embedding(node_features)
        edge_features_updated = self.edges_initial_embedding(edge_features)
        global_features_updated = self.global_initial_embedding(global_features)

        for i in range(g2g_num):
            node_features_updated, edge_features_updated, global_features_updated = self.g2g_module(node_features_updated,
                                                                                                    edge_index,
                                                                                                    edge_features_updated,
                                                                                                    global_features_updated)

        features_concatenated = self.features_set2set.transform_then_concat(node_features_updated,
                                                                            edge_features_updated,
                                                                            global_features_updated)

        predicted_value = self.g2p_module(features_concatenated)

        return predicted_value

# adapted from https://stackoverflow.com/questions/71998978/early-stopping-in-pytorch
class EarlyStopper:
    def __init__(self, patience=150, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

def train_loop(dataloader, dataloader2, model, loss_fn, optimizer):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # print("before size")
    size = len(dataloader.dataset)
    # print("after size")
    # print(size)
    loss_batch = []
    for batch, (X1, X2, X3, X4, y) in enumerate(dataloader.dataset):
        X1, X2, X3, X4, y = X1.to(device), X2.to(device), X3.to(device), X4.to(device), y.to(device)
        pred = model(X1.float(), X2.float(), X3.float(), X4.float())
        yReshaped = torch.Tensor([y]).to(device)
        # print(yReshaped.shape)
        # print("Prediction: %s, Actual value %s", pred, yReshaped)
        loss = loss_fn(pred, yReshaped)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_batch.append(loss.detach().item())

    loss_epoch = np.average(loss_batch)

    print("Training loss: ", loss_epoch)

    val_loss_batch = []

    with torch.no_grad():
        for batch, (X1, X2, X3, X4, y) in enumerate(dataloader2.dataset):
            X1, X2, X3, X4, y = X1.to(device), X2.to(device), X3.to(device), X4.to(device), y.to(device)
            pred = model(X1.float(), X2.float(), X3.float(), X4.float())
            yReshaped = torch.Tensor([y]).to(device)
            loss = loss_fn(pred, yReshaped)

            val_loss_batch.append(loss.detach().item())

    val_loss_epoch = np.average(val_loss_batch)

    print("Validation loss: ", val_loss_epoch)

    return loss_epoch, val_loss_epoch


'''
def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader.dataset:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    # print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
'''


def plotLearningCurves(train_loss, val_loss):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(len(train_loss))
    print(len(val_loss))

    print("train_loss: ", train_loss)
    print("val_loss: ", val_loss)

    train_loss = torch.Tensor(train_loss).to(device)
    val_loss = torch.Tensor(val_loss).to(device)

    plt.figure(figsize=(10, 5))
    plt.title("Training and Validation Loss")
    plt.xticks(range(1, len(train_loss) + 1))
    plt.plot([i + 1 for i in range(len(train_loss))], val_loss, label="val")
    plt.plot([i + 1 for i in range(len(train_loss))], train_loss, label="train")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


def runGraphNeuralNetwork():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # print(f"Using {device} device")

    model = GraphNeuralNetwork()
    model = model.to(device).float()

    learning_rate = 1e-3
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_loss = []
    val_loss = []
    loss_epochs = []
    val_loss_epochs = []

    lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50)

    print("we want to train the model!")

    test_data_loader, val_data_loader, train_data_loader = train_val_test_split()

    early_stopper = EarlyStopper(patience=150, min_delta=0.0)

    epochs = 1000

    for t in range(epochs):
        print(f"Epoch {t + 1}")
        print("Learning rate: ", optimizer.param_groups[0]["lr"])
        print("-------------------------------")
        # we need to check to see if train_data and val_data is being shuffled before each epoch along with playing around with different initializations (and can do multiple reruns)
        # we can also try SGD for a few epochs (5) before doing Adam or maybe try SGD for all 20 epochs
        # we can run several jupyter notebooks in parallel
        train_loss_epoch_value, val_loss_epoch_value = train_loop(train_data_loader.dataset, val_data_loader.dataset,
                                                                  model, loss_fn, optimizer)
        train_loss.append(train_loss_epoch_value)
        val_loss.append(val_loss_epoch_value)

        # test_loop(test_data, model, loss_fn)

        lr_scheduler.step(val_loss_epoch_value)

        if early_stopper.early_stop(val_loss_epoch_value):
            break

    print("Done!")

    plotLearningCurves(train_loss, val_loss)


def main():
    # torch.autograd.set_detect_anomaly(True)
    runGraphNeuralNetwork()


if __name__ == '__main__':
    main()