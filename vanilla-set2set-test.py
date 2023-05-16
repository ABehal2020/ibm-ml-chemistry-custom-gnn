from torch_geometric.nn.aggr import Set2Set
import torch

inputFeatures = torch.Tensor([[1, 2, 3],
                             [4, 5, 6],
                             [7, 8, 9]])

s2s = Set2Set(in_channels=inputFeatures.shape[1], processing_steps=6)

outputFeatures = s2s(inputFeatures)

print(outputFeatures)