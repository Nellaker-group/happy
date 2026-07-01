import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GraphConv, norm


class GraphSAINT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else hidden_channels
            hidden_channels = out_channels if i == num_layers - 1 else hidden_channels
            self.convs.append(GraphConv(in_channels, hidden_channels))
            self.bns.append(norm.BatchNorm(hidden_channels))

    def set_aggr(self, aggr):
        for conv in self.convs:
            conv.aggr = aggr

    def forward(self, x, edge_index, edge_weight=None):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index, edge_weight)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index, edge_weight)
        return torch.log_softmax(x, dim=-1)

    def inference(self, x_all, subgraph_loader, device):
        for i, conv in enumerate(self.convs):
            xs = []
            for batch in subgraph_loader:
                x = x_all[batch.n_id.to(x_all.device)].to(device)
                x = conv(x, batch.edge_index.to(device))
                if i != len(self.convs) - 1:
                    x = self.bns[i](x)
                    x = F.relu(x)
                xs.append(x[:batch.batch_size].cpu())
            x_all = torch.cat(xs, dim=0)

            if i == self.num_layers - 2:
                embeddings = x_all.detach().clone()

        return x_all, embeddings
