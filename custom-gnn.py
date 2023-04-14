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
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import dense_to_sparse, add_self_loops, to_scipy_sparse_matrix
from torch_geometric.data import Data
import torch.nn.functional as F
from six.moves import urllib
import deepchem as dc
import random
from dgl.nn import Set2Set
import multiprocessing
from torchvision.transforms import Compose, ToTensor

def get_data():
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    np.random.seed(0)
    random.seed(0)

    warnings.filterwarnings("ignore")
    sns.set_context("notebook")
    sns.set_style(
        "dark",
        {
            "xtick.bottom": True,
            "ytick.left": True,
            "xtick.color": "#666666",
            "ytick.color": "#666666",
            "axes.edgecolor": "#666666",
            "axes.linewidth": 0.8,
            "figure.dpi": 300,
        },
    )
    color_cycle = ["#1BBC9B", "#F06060", "#5C4B51", "#F3B562", "#6e5687"]
    mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=color_cycle)

    opener = urllib.request.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0')]
    urllib.request.install_opener(opener)

    # had to rehost because dataverse isn't reliable
    soldata = pd.read_csv(
        "https://github.com/whitead/dmol-book/raw/main/data/curated-solubility-dataset.csv"
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
    for i in range(len(soldata)):
        graphInstance = gen_smiles2graph(soldata.SMILES[i])
        if hasattr(graphInstance, "node_features") and hasattr(graphInstance, "edge_index") and hasattr(graphInstance, "edge_features"):
            graph.append(graphInstance)
            sol.append(soldata.Solubility[i])

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

        print("node_features: ", type(graphInstance.node_features))
        print("edge_index: ", type(graphInstance.edge_index))
        print("edge_features: ", type(graphInstance.edge_features))

        if self.transform:
            graphInstance = dc.feat.GraphData(self.transform(graphInstance.node_features), self.transform(graphInstance.edge_index), self.transform(graphInstance.edge_features))
        if self.target_transform:
            solInstance = self.target_transform(solInstance)
        return graphInstance, solInstance

def train_val_test_split():
    graph, sol = featurize_data()

    dataset = CustomDataset(graphAll=graph, solAll=sol, transform=Compose([ToTensor()]), target_transform=Compose([ToTensor()]))

    cores = multiprocessing.cpu_count() # Count the number of cores in a computer
    # batch_size=1 was original default - 100 was used in bondnet paper
    # set num_workers=cores for best performance
    dataloader = data.DataLoader(dataset, batch_size=1,
                            shuffle=True, num_workers=1)
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

    test_data, val_data, train_data = data.random_split(dataloader, [test_data_size, val_data_size, train_data_size], generator=torch.Generator().manual_seed(0))

    # set num_workers=cores for best performance
    test_data_loader = data.DataLoader(test_data, batch_size=1, shuffle=True, num_workers=1)
    val_data_loader = data.DataLoader(val_data, batch_size=1, shuffle=True, num_workers=1)
    train_data_loader = data.DataLoader(train_data, batch_size=1, shuffle=True, num_workers=1)
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
        torch.manual_seed(0)
        self.fc_initial_embedding = nn.Linear(c_in, c_out)
    
    def forward(self, features):
        features = self.fc(features)
        features = F.relu(features)
        
        return features

# neural network with two fully connected layers
class FCNN(torch.nn.Module):
    # c_in1 = 24, c_out1 = 256, c_out2 = 24
    def __init__(self, c_in1, c_out1, c_out2):
        super().__init__()
        torch.manual_seed(0)
        self.fc1 = nn.Linear(c_in1, c_out1)
        torch.manual_seed(0)
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

# implementation of equation 5 in bondnet paper 
# https://pubs.rsc.org/en/content/articlepdf/2021/sc/d0sc05251e
class NodeFeatures(torch.nn.Module):
    # c_in1 = 24, c_out1 = 24, c_out2 = 24
    def __init__(self, c_in1, c_out1, c_out2):
        super().__init__()
        # self.fc_initial_embedding = InitialEmbedding(c_in=30, c_out=24)
        self.FCNN_one = FCNN(c_in1=c_in1, c_out1=c_out1, c_out2=c_out2)
        self.FCNN_two = FCNN(c_in1=c_in1, c_out1=c_out1, c_out2=c_out2)
        
    def forward(self, node_features, edge_index, edge_features):
        sigmoidFunction = torch.nn.Sigmoid()
        
        original_node_features = node_features.detach().clone()
        
        epsilon = 1e-7
        
        for i in range(len(node_features)):
            # DOUBLE CHECK WITH DAS
            # intermediate_node_feature = self.FCNN_one(node_features[i].T)
            intermediate_node_feature = self.FCNN_one(original_node_features[i].T)
            
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
            
            for j in range(len(edge_index[0])):
                if edge_index[0][j] == i:
                    other_nodes_indices.append(int(edge_index[1][j]))
                    other_edges_indices.append(j)
                if edge_index[1][j] == i:
                    other_nodes_indices.append(int(edge_index[0][j]))
                    other_edges_indices.append(j)
            
            '''
            print("current node index: ", i)
            print("other_nodes_indices: ", other_nodes_indices)
            print("other_edges_indices: ", other_edges_indices)
            '''
            
            for other_edge_index in other_edges_indices:
                # print("SIGMOID ALERT TEST TEST TEST: ", sigmoidFunction(edge_features[other_edge_index]))
                other_edges_numerators.append(sigmoidFunction(edge_features[other_edge_index]))
                other_edges_denominator += sigmoidFunction(edge_features[other_edge_index])
                
            # print("other_edges_numerators: ", other_edges_numerators)
            # print("other_edges_denominator: ", other_edges_denominator)
            
            for other_edge_numerator, other_node_index in zip(other_edges_numerators, other_nodes_indices):
                edge_hat = other_edge_numerator/other_edges_denominator
                # DOUBLE CHECK WITH DAS
                # other_node_updated = self.FCNN_two(node_features[other_node_index].T) 
                other_node_updated = self.FCNN_two(original_node_features[other_node_index].T) 
                intermediate_node_feature += edge_hat * other_node_updated
                
                # print("edge_hat: ", edge_hat)
                '''
                print("node_features[other_node_index].T: ", node_features[other_node_index].T)
                print("node_features[other_node_index].T.size: ", node_features[other_node_index].T.size())
                print("other_node_updated: ", other_node_updated)
                print("other_node_updated.size: ", other_node_updated.size())
                '''
                
            print("intermediate_node_feature: ", intermediate_node_feature)
            print("intermediate_node_feature.size: ", intermediate_node_feature.size())
                
            '''
            print("reLuOutput: ", F.relu(intermediate_node_feature))
            print("reLuOutput.size: ", F.relu(intermediate_node_feature).size())
            print("original_node_features[i].T", original_node_features[i].T)
            print("original_node_features[i].T.size", original_node_features[i].T.size())
            print("calculated updated node_features[i]: ", (original_node_features[i].T + F.relu(intermediate_node_feature)).T)
            print("calculated updated node_features[i].size(): ", (original_node_features[i].T + F.relu(intermediate_node_feature)).T.size())
            '''
            
            # UPDATE TO INCLUDE THIS AT SOME POINT
            # intermediate_node_feature --> batch normalization --> drop out --> then ReLu
            # should I use batch norm 1D and what should my feature size be at this point?
            '''
            batchNorm1dLayer = nn.BatchNorm1d(intermediate_node_feature.size(dim=0))
            dropoutLayer = nn.Dropout(p=0.1)
            
            intermediate_node_feature = batchNorm1dLayer(torch.reshape(intermediate_node_feature, (1, intermediate_node_feature.size(dim=0))))
            intermediate_node_feature = torch.reshape(intermediate_node_feature, (-1,))
            intermediate_node_feature = dropoutLayer(intermediate_node_feature)
            '''
            
            instanceNorm1dLayer = nn.InstanceNorm1d(intermediate_node_feature.size(dim=0))
            dropoutLayer = nn.Dropout(p=0.1)
            
            intermediate_node_feature = instanceNorm1dLayer(torch.reshape(intermediate_node_feature, (1, intermediate_node_feature.size(dim=0))))
            intermediate_node_feature = torch.reshape(intermediate_node_feature, (-1,))
            intermediate_node_feature = dropoutLayer(intermediate_node_feature)
            
            # node_features[i] = F.relu(intermediate_node_feature).T
            node_features[i] = (original_node_features[i].T + F.relu(intermediate_node_feature)).T
            
            print("actually updated node_features[i]: ", node_features[i])
            print("actually updated node_features[i].size(): ", node_features[i].size())
            print("********** NODE UPDATED SUCCESSFULLY ****************")
            
        return node_features
        
# implementation of equation 4 in bondnet paper
# https://pubs.rsc.org/en/content/articlepdf/2021/sc/d0sc05251e
class EdgeFeatures(torch.nn.Module):
    # c_in1 = 24, c_out1 = 24, c_out2 = 24
    def __init__(self, c_in1, c_out1, c_out2):
        super().__init__()
        # self.fc_initial_embedding = InitialEmbedding(c_in=11, c_out=24)
        self.FCNN_one = FCNN(c_in1=c_in1, c_out1=c_out1, c_out2=c_out2)
        self.FCNN_two = FCNN(c_in1=c_in1, c_out1=c_out1, c_out2=c_out2)
        
    def forward(self, node_features, edge_index, edge_features):
        original_edge_features = edge_features.detach().clone()
        
        for i in range(len(edge_index[0])):
            # summing node features involved in the given edge and transforming them
            firstNodeIndex = int(edge_index[0][i])
            secondNodeIndex = int(edge_index[1][i])
            node_features_sum = node_features[firstNodeIndex] + node_features[secondNodeIndex]
            intermediate_node_features = self.FCNN_one(node_features_sum.T)
            
            print("firstNodeIndex: ", firstNodeIndex)
            print("secondNodeIndex: ", secondNodeIndex)
            print("node_features[firstNodeIndex]: ", node_features[firstNodeIndex])
            print("node_features[secondNodeIndex]: ", node_features[secondNodeIndex])
            print("node_features_sum: ", node_features_sum)
            print("node_features_sum.size: ", node_features_sum.size())
            print("node_features_sum.T: ", node_features_sum.T)
            print("node_features_sum.T.size: ", node_features_sum.T.size())
            print("intermediate_node_features: ", intermediate_node_features)
            print("intermediate_node_features.size: ", intermediate_node_features.size())
            
            # transforming the features of the given edge 
            intermediate_edge_feature = self.FCNN_two(edge_features[i].T)
            
            print("edge_features index: ", i)
            print("edge_features: ", edge_features[i])
            print("edge_features.size: ", edge_features[i].size())
            print("edge_features.T: ", edge_features[i].T)
            print("edge_features.T.size(): ", edge_features[i].T.size())
            print("intermediate_edge_feature: ", intermediate_edge_feature)
            print("intermediate_edge_feature.size: ", intermediate_edge_feature.size())
            print("intermediate_edge_feature.size dim 0: ", intermediate_edge_feature.size(dim=0))

            # merging node features with features of the given edge
            
            # UPDATE TO INCLUDE THIS AT SOME POINT
            # intermediate_node_features + intermediate_edge_feature --> batch normalization --> drop out --> then ReLu
            
            intermediate_features_relu_input = intermediate_node_features + intermediate_edge_feature
            
            instanceNorm1dLayer = nn.InstanceNorm1d(intermediate_features_relu_input.size(dim=0))
            dropoutLayer = nn.Dropout(p=0.1)
            
            intermediate_features_relu_input = instanceNorm1dLayer(torch.reshape(intermediate_features_relu_input, (1, intermediate_features_relu_input.size(dim=0))))
            intermediate_features_relu_input = torch.reshape(intermediate_features_relu_input, (-1,))                                                              
            intermediate_features_relu_input = dropoutLayer(intermediate_features_relu_input)
            
            intermediate_features = F.relu(intermediate_features_relu_input)
            
            print("intermediate_features: ", intermediate_features)
            print("intermediate_features.size: ", intermediate_features.size())
            print("original_edge_features[i].T: ", original_edge_features[i].T)
            print("calculated updated edge_features[i]: ", (original_edge_features[i].T + intermediate_features).T)
            
            # updating edge features
            edge_features[i] = (original_edge_features[i].T + intermediate_features).T
            
            print("actually updated edge_features[i]: ", edge_features[i])
            print("********** EDGE UPDATED SUCCESSFULLY ****************")
            
        return edge_features
    
class Graph2Graph(torch.nn.Module):
    def __init__(self, c_in1, c_out1, c_out2):
        super().__init__()
        self.NodeFeaturesConvolution = NodeFeatures(c_in1, c_out1, c_out2)
        self.EdgeFeaturesConvolution = EdgeFeatures(c_in1, c_out1, c_out2)
        
    def forward(self, node_features, edge_index, edge_features):
        node_features = self.NodeFeaturesConvolution
        edge_features = self.EdgeFeaturesConvolution
        
        return node_features, edge_features
    
class Features_Set2Set():
    def __init__(self, initial_dim_out):
        self.node_s2s = Set2Set(initial_dim_out, 6, 3)
        self.edge_s2s = Set2Set(initial_dim_out, 6, 3)
    
    def transform_then_concat(self, node_features, edge_index, edge_features):
        deepchem_graph = dc.feat.GraphData(node_features, edge_index, edge_features)
        dgl_graph = deepchem_graph.to_dgl_graph()
        node_features_transformed = self.node_s2s(dgl_graph, node_features)
        edge_features_transformed = self.edge_s2s(dgl_graph, edge_features)
        
        return torch.cat(node_features_transformed, edge_features_transformed)

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
    def __init__(self, nodes_initial_dim_in=30, edges_initial_dim_in=11, initial_dim_out=24, g2g_hidden_dim=256, g2p_dim_1=256, g2p_dim_2=128, g2p_dim_3=64):
        super(GraphNeuralNetwork, self).__init__()
        self.nodes_initial_embedding = InitialEmbedding(nodes_initial_dim_in, initial_dim_out)
        self.edges_initial_embedding = InitialEmbedding(edges_initial_dim_in, initial_dim_out)
        self.g2g_module = Graph2Graph(initial_dim_out, g2g_hidden_dim, initial_dim_out)
        self.features_set2set = Features_Set2Set(initial_dim_out)
        self.g2p_module = Graph2Property(initial_dim_out, g2p_dim_1, g2p_dim_2, g2p_dim_3, 1)
        
    def forward(self, graph_instance, g2g_num=4):
        node_features = graph_instance.node_features 
        edge_index = graph_instance.edge_index
        edge_features = graph_instance.edge_features
        
        node_features_updated = self.nodes_initial_embedding(node_features)
        edge_features_updated = self.edges_initial_embedding(edge_features)
        
        for i in range(g2g_num):
            node_features_updated, edge_features_updated = self.g2g_module(node_features, edge_index, edge_features)
            
        features_concatenated = Features_Set2Set(node_features_updated, edge_index, edge_features_updated)
        
        predicted_value = self.g2p_module(features_concatenated)
        
        return predicted_value

def train_loop(dataloader, dataloader2, model, loss_fn, optimizer):
    # print("before size")
    size = len(dataloader.dataset)
    # print("after size")
    # print(size)
    loss_batch = []
    for batch, (X, y) in enumerate(dataloader.dataset):
        pred = model(X)
        yReshaped = torch.Tensor([y]).reshape(1, 1, 1)
        # print(yReshaped.shape)
        # print("Prediction: %s, Actual value %s", pred, yReshaped)
        loss = loss_fn(pred, yReshaped)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_batch.append(loss.item())

    loss_epoch = np.average(loss_batch)

    print("Training loss: ", loss_epoch)

    val_loss_batch = []

    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader2.dataset):
            pred = model(X)
            yReshaped = torch.Tensor([y]).reshape(1, 1, 1)
            loss = loss_fn(pred, yReshaped)
      
            val_loss_batch.append(loss.item())
    
    val_loss_epoch = np.average(val_loss_batch)

    print("Validation loss: ", val_loss_epoch)

    return loss_epoch, val_loss_epoch

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

def runGraphNeuralNetwork():
    model = GraphNeuralNetwork()

    learning_rate = 1e-3
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_loss = []
    val_loss = []
    loss_epochs = []
    val_loss_epochs = []

    print("we want to train the model!")

    test_data_loader, val_data_loader, train_data_loader = train_val_test_split()

    epochs = 20
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        # we need to check to see if train_data and val_data is being shuffled before each epoch along with playing around with different initializations (and can do multiple reruns)
        # we can also try SGD for a few epochs (5) before doing Adam or maybe try SGD for all 20 epochs
        # we can run several jupyter notebooks in parallel
        train_loss_epoch_value, val_loss_epoch_value = train_loop(train_data_loader.dataset, val_data_loader.dataset,
                                                                  model, loss_fn, optimizer)
        train_loss.append(train_loss_epoch_value)
        val_loss.append(val_loss_epoch_value)
        # test_loop(test_data, model, loss_fn)
    print("Done!")

def main():
    runGraphNeuralNetwork()

if __name__ == '__main__':
    main()

'''
print("let's see what's coming out of our dataloader")

xNodeFError = 0
xEdgeIError = 0
xEdgeFError = 0

for batch, (X, y) in enumerate(dataloader.dataset):
    if not hasattr(X, "node_features"):
        xNodeFError += 1
    if not hasattr(X, "edge_index"):
        xEdgeIError += 1
    if not hasattr(X, "edge_features"):
        xEdgeFError += 1

print("xNodeError: ", xNodeFError)
print("xEdgeIError: ", xEdgeIError)
print("xEdgeFError: ", xEdgeFError)
'''