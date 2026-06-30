import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GATv2Conv


class GAT(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        heads,
        num_layers,
        dropout=0.0,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.dropout = dropout

        for i in range(num_layers):
            if i == 0:
                self.convs.append(
                    GATConv(
                        in_channels,
                        hidden_channels,
                        heads,
                        dropout=dropout,
                        add_self_loops=False,
                    )
                )
            self.convs.append(
                GATConv(
                    hidden_channels * heads,
                    hidden_channels,
                    heads,
                    dropout=dropout,
                    add_self_loops=False,
                )
            )
            if i == num_layers - 1:
                self.convs.append(
                    GATConv(
                        hidden_channels * heads,
                        out_channels,
                        heads=heads,
                        concat=False,
                        dropout=dropout,
                        add_self_loops=False,
                    )
                )

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = conv(x, edge_index)
            if i != len(self.convs) - 1:
                x = F.relu(x)
        return F.log_softmax(x, dim=-1)

    def inference(self, x_all, subgraph_loader, device):
        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch.
        for i, conv in enumerate(self.convs):
            xs = []
            for batch in subgraph_loader:
                x = x_all[batch.n_id.to(x_all.device)].to(device)
                x = conv(x, batch.edge_index.to(device))
                if i != len(self.convs) - 1:
                    x = F.relu(x)
                xs.append(x[:batch.batch_size].cpu())

            x_all = torch.cat(xs, dim=0)

            if i == self.num_layers - 2:
                embeddings = x_all.detach().clone()

        return x_all, embeddings


class GATv2(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        heads,
        num_layers,
        dropout=0.0,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.dropout = dropout

        for i in range(num_layers):
            if i == 0:
                self.convs.append(
                    GATv2Conv(
                        in_channels,
                        hidden_channels,
                        heads,
                        dropout=dropout,
                        add_self_loops=False,
                        share_weights=True,
                    )
                )
            self.convs.append(
                GATv2Conv(
                    hidden_channels * heads,
                    hidden_channels,
                    heads,
                    dropout=dropout,
                    add_self_loops=False,
                    share_weights=True,
                )
            )
            if i == num_layers - 1:
                self.convs.append(
                    GATv2Conv(
                        hidden_channels * heads,
                        out_channels,
                        heads=heads,
                        concat=False,
                        dropout=dropout,
                        add_self_loops=False,
                        share_weights=True,
                    )
                )

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = conv(x, edge_index)
            if i != len(self.convs) - 1:
                x = F.relu(x)
        return F.log_softmax(x, dim=-1)

    def inference(self, x_all, subgraph_loader, device):
        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch.
        for i, conv in enumerate(self.convs):
            xs = []
            for batch in subgraph_loader:
                x = x_all[batch.n_id.to(x_all.device)].to(device)
                x = conv(x, batch.edge_index.to(device))
                if i != len(self.convs) - 1:
                    x = F.relu(x)
                xs.append(x[:batch.batch_size].cpu())

            x_all = torch.cat(xs, dim=0)

            if i == self.num_layers - 2:
                embeddings = x_all.detach().clone()

        return x_all, embeddings
