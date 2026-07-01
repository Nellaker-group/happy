import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, TopKPooling
from torch_geometric.typing import SparseTensor
from torch_geometric.utils import add_self_loops, remove_self_loops
from torch_geometric.utils.repeat import repeat

class GraphUNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout, num_layers, k=2):
        super(GraphUNet, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.feature_skip = True

        self.down_convs = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.down_convs.append(GCNConv(in_channels, hidden_channels, improved=True))

        for i in range(num_layers):
            self.pools.append(TopKPooling(hidden_channels, 1/k))
            self.down_convs.append(GCNConv(hidden_channels, hidden_channels, improved=True))

        self.up_convs = nn.ModuleList()

        for i in range(num_layers - 1):
            self.up_convs.append(GCNConv(hidden_channels, hidden_channels, improved=True))

        self.up_convs.append(GCNConv(hidden_channels, out_channels, improved=True))

    def forward(self, x, edge_index, batch=None):
        edge_weight = x.new_ones(edge_index.size(1))

        x = self.down_convs[0](x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        xs = [x]
        edge_indices = [edge_index]
        edge_weights = [edge_weight]
        perms = []

        for i in range(1, self.num_layers + 1):
            edge_index, edge_weight = self.augment_adj(edge_index, edge_weight, x.size(0))
            x, edge_index, edge_weight, batch, perm, _ = self.pools[i - 1](x, edge_index, edge_weight, batch)

            x = self.down_convs[i](x, edge_index, edge_weight)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

            if i < self.num_layers:
                xs += [x]
                edge_indices += [edge_index]
                edge_weights += [edge_weight]

            perms += [perm]

        for i in range(self.num_layers):
            j = self.num_layers - 1 - i

            res = xs[j]
            edge_index = edge_indices[j]
            edge_weight = edge_weights[j]
            perm = perms[j]

            up = torch.zeros_like(res)
            up[perm] = x
            if self.feature_skip:
                x = res + up
            else:
                rand = torch.rand_like(res)
                x = 0.001 * rand + up

            x = self.up_convs[i](x, edge_index, edge_weight)
            x = F.relu(x) if i < self.num_layers - 1 else x

        return F.log_softmax(x, dim=1)
    
    def inference(self, x, edge_index, batch=None):
        edge_weight = x.new_ones(edge_index.size(1))

        x = self.down_convs[0](x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        xs = [x]
        edge_indices = [edge_index]
        edge_weights = [edge_weight]
        perms = []

        for i in range(1, self.num_layers + 1):
            edge_index, edge_weight = self.augment_adj(edge_index, edge_weight, x.size(0))
            x, edge_index, edge_weight, batch, perm, _ = self.pools[i - 1](x, edge_index, edge_weight, batch)

            x = self.down_convs[i](x, edge_index, edge_weight)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

            if i < self.num_layers:
                xs += [x]
                edge_indices += [edge_index]
                edge_weights += [edge_weight]

            perms += [perm]

        for i in range(self.num_layers):
            j = self.num_layers - 1 - i

            res = xs[j]
            edge_index = edge_indices[j]
            edge_weight = edge_weights[j]
            perm = perms[j]

            up = torch.zeros_like(res)
            
            up[perm] = x
            if self.feature_skip:
                x = res + up
            else:
                rand = torch.rand_like(res)
                x = 0.001 * rand + up

            x = self.up_convs[i](x, edge_index, edge_weight)
            x = F.relu(x) if i < self.num_layers - 1 else x

        return x
    

    def augment_adj(self, edge_index, edge_weight, num_nodes):
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        edge_index, edge_weight = add_self_loops(edge_index, edge_weight, num_nodes=num_nodes)
        adj = SparseTensor.from_edge_index(edge_index, edge_weight, sparse_sizes=(num_nodes, num_nodes))
        adj = adj @ adj
        row, col, edge_weight = adj.coo()
        edge_index = torch.stack([row, col], dim=0)
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        return edge_index, edge_weight
    

class GraphUNetNoSkip(GraphUNet):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout, num_layers, k=2):
        super(GraphUNetNoSkip, self).__init__(in_channels, hidden_channels, out_channels, dropout, num_layers, k)
        self.feature_skip = False