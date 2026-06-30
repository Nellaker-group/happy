import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, global_mean_pool


class ShaDowGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.convs = nn.ModuleList()

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else hidden_channels
            self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.lin = torch.nn.Linear(2 * hidden_channels, out_channels)

    def forward(self, x, edge_index, batch, root_n_id):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index).relu()
            x = F.dropout(x, p=self.dropout, training=self.training)
        # We merge both central node embeddings and subgraph embeddings:
        x = torch.cat([x[root_n_id], global_mean_pool(x, batch)], dim=-1)
        x = self.lin(x)
        return F.log_softmax(x, dim=-1)

    def inference(self, x, edge_index, batch, root_n_id):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index).relu()
        # We merge both central node embeddings and subgraph embeddings:
        x = torch.cat([x[root_n_id], global_mean_pool(x, batch)], dim=-1)
        embeddings = x.detach().clone()
        x = self.lin(x)
        return x, embeddings
