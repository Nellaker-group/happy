import torch.nn as nn
import torch.nn.functional as F
import torch
from torch_geometric.nn import SAGEConv, norm


class ClusterGCN(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        dropout,
        num_layers,
        reduce_dims=None,
    ):
        super(ClusterGCN, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.reduce_dims = reduce_dims
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else hidden_channels
            if i == num_layers - 1 and reduce_dims is None:
                hidden_channels = out_channels
            else:
                hidden_channels = hidden_channels
            self.convs.append(SAGEConv(in_channels, hidden_channels))
            self.bns.append(norm.BatchNorm(hidden_channels))

        if reduce_dims is not None:
            self.lin1 = nn.Linear(hidden_channels, reduce_dims)
            self.lin2 = nn.Linear(reduce_dims, out_channels)

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i == len(self.convs) - 1 and self.reduce_dims is None:
                continue
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        if self.reduce_dims is not None:
            x = self.lin1(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.lin2(x)
        return F.log_softmax(x, dim=-1)

    def inference(self, x_all, subgraph_loader, device):
        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch.
        for i, conv in enumerate(self.convs):
            xs = []
            for batch_size, n_id, adj in subgraph_loader:
                edge_index, _, size = adj.to(device)
                x = x_all[n_id].to(device)
                x_target = x[:size[1]]
                x = conv((x, x_target), edge_index)
                if i != len(self.convs) - 1 or self.reduce_dims is not None:
                    x = self.bns[i](x)
                    x = F.relu(x)
                xs.append(x)
            x_all = torch.cat(xs, dim=0)

            if i == self.num_layers - 2 and self.reduce_dims is None:
                embeddings = x_all.detach().cpu().clone()

        if self.reduce_dims is not None:
            x_all = self.lin1(x_all)
            x_all = F.relu(x_all)
            embeddings = x_all.detach().cpu().clone()
            x_all = self.lin2(x_all)

        return x_all.cpu(), embeddings
