import torch.nn as nn
import torch.nn.functional as F
import torch
from torch_geometric.nn import SAGEConv, ClusterGCNConv, norm, JumpingKnowledge

from happy.models.utils.custom_layers import WeightedSAGEConv


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
            self.lin1 = nn.Linear(hidden_channels, int(hidden_channels / 2))
            self.lin_bn1 = nn.BatchNorm1d(int(hidden_channels / 2))
            self.lin2 = nn.Linear(int(hidden_channels / 2), reduce_dims)
            self.lin_bn2 = nn.BatchNorm1d(reduce_dims)
            self.lin3 = nn.Linear(reduce_dims, out_channels)

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
            x = self.lin_bn1(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.lin2(x)
            x = self.lin_bn2(x)
            x = self.lin3(x)
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
            x_all = self.lin_bn1(x_all)
            x_all = F.relu(x_all)
            x_all = self.lin2(x_all)
            x_all = self.lin_bn2(x_all)
            embeddings = x_all.detach().cpu().clone()
            x_all = self.lin3(x_all)

        return x_all.cpu(), embeddings


class JumpingClusterGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout, num_layers):
        super(JumpingClusterGCN, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.jump = JumpingKnowledge(mode="cat")
        self.lin1 = nn.Linear(num_layers * hidden_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, out_channels)

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else hidden_channels
            self.convs.append(SAGEConv(in_channels, hidden_channels))
            self.bns.append(norm.BatchNorm(hidden_channels))

    def forward(self, x, edge_index):
        xs = []
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            xs.append(x)
        x = self.jump(xs)
        x = self.lin1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)

    def inference(self, x_all, subgraph_loader, device):
        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch.
        xs = []
        for i, conv in enumerate(self.convs):
            batch_xs = []
            for batch_size, n_id, adj in subgraph_loader:
                edge_index, _, size = adj.to(device)
                x = x_all[n_id].to(device)
                x_target = x[: size[1]]
                x = conv((x, x_target), edge_index)
                x = self.bns[i](x)
                x = F.relu(x)
                batch_xs.append(x)
            x_all = torch.cat(batch_xs, dim=0)
            xs.append(x_all)

        x = self.jump(xs)
        x = self.lin1(x)
        embeddings = x.detach().cpu().clone()
        x = F.relu(x)
        x = self.lin2(x)
        return x.cpu(), embeddings


class ClusterGCNConvNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super(ClusterGCNConvNet, self).__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList()

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else hidden_channels
            hidden_channels = out_channels if i == num_layers - 1 else hidden_channels
            self.convs.append(
                ClusterGCNConv(in_channels, hidden_channels, add_self_loops=False)
            )

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)
        return F.log_softmax(x, dim=-1)

    def inference(self, x_all, subgraph_loader, device):
        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch:
        for i, conv in enumerate(self.convs):
            xs = []
            for batch in subgraph_loader:
                x = x_all[batch.n_id.to(x_all.device)].to(device)
                x = conv(x, batch.edge_index.to(device))
                if i < len(self.convs) - 1:
                    x = x.relu_()
                xs.append(x[: batch.batch_size].cpu())
            x_all = torch.cat(xs, dim=0)

            if i == self.num_layers - 2:
                embeddings = x_all.detach().clone()

        return x_all, embeddings


class ClusterGCNEdges(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        dropout,
        num_layers,
        reduce_dims=None,
    ):
        super(ClusterGCNEdges, self).__init__()
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
            self.convs.append(WeightedSAGEConv(in_channels, hidden_channels))
            self.bns.append(norm.BatchNorm(hidden_channels))

        if reduce_dims is not None:
            self.lin1 = nn.Linear(hidden_channels, reduce_dims)
            self.lin2 = nn.Linear(reduce_dims, out_channels)

    def forward(self, x, edge_index, edge_attr):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_attr)
            if i == len(self.convs) - 1 and self.reduce_dims is None:
                continue
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        if self.reduce_dims is not None:
            x = self.lin1(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.lin2(x)
        return F.log_softmax(x, dim=-1)

    def inference(self, x_all, edge_attr, subgraph_loader, device):
        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch.
        edge_attr = edge_attr.to(device)
        for i, conv in enumerate(self.convs):
            xs = []
            for batch_size, n_id, adj in subgraph_loader:
                edge_index, e_id, size = adj.to(device)
                x = x_all[n_id].to(device)
                edge_attr_batch = edge_attr[e_id].to(device)
                x_target = x[:size[1]]
                x = conv((x, x_target), edge_index, edge_attr_batch)
                if i != len(self.convs) - 1 or self.reduce_dims is not None:
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


class ClusterGCNMultilabel(ClusterGCN):
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
            x = self.lin_bn1(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.lin2(x)
            x = self.lin_bn2(x)
            x = self.lin3(x)
        return torch.sigmoid(x)
    
class ClusterGCNMultihead(ClusterGCN):
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
            x = self.lin_bn1(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.lin2(x)
            x = self.lin_bn2(x)
            x = self.lin3(x)

        x1 = F.log_softmax(x[:, :-2], dim=-1)
        x2 = F.log_softmax(x[:, -2:], dim=-1)

        return x1, x2
