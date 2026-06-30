import torch.nn as nn
import torch.nn.functional as F
import torch
from torch_geometric.nn import SAGEConv, dense_diff_pool, GraphConv, TopKPooling, norm


class SAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers):
        super(SAGE, self).__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            in_channels = in_channels if i == 0 else hidden_channels
            self.convs.append(SAGEConv(in_channels, hidden_channels))

    def forward(self, x, adjs):
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[: size[1]]  # Target nodes are always placed first.
            x = self.convs[i]((x, x_target), edge_index)
            if i != self.num_layers - 1:
                x = x.relu()
                x = F.dropout(x, p=0.5, training=self.training)
        return x

    def full_forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != self.num_layers - 1:
                x = x.relu()
                x = F.dropout(x, p=0.5, training=self.training)
        return x


class SupervisedSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout, num_layers):
        super(SupervisedSAGE, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else hidden_channels
            hidden_channels = out_channels if i == num_layers - 1 else hidden_channels
            self.convs.append(SAGEConv(in_channels, hidden_channels))
            self.bns.append(norm.BatchNorm(hidden_channels))

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = self.bns[i](x)
                x = x.relu_()
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x

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
                    x = self.bns[i](x)
                    x = x.relu_()
                xs.append(x[:batch.batch_size].cpu())
            x_all = torch.cat(xs, dim=0)

            if i == self.num_layers - 2:
                embeddings = x_all.detach().clone()

        return x_all, embeddings


class SageForDiffPool(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super(SageForDiffPool, self).__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else hidden_channels
            hidden_channels = out_channels if i == num_layers - 1 else hidden_channels
            self.convs.append(SAGEConv(in_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))

    def forward(self, x, adjs):
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[: size[1]]  # Target nodes are always placed first.
            x = self.bns[i](F.relu(self.convs[i]((x, x_target), edge_index)))
        return x


class SupervisedDiffPool(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_sage_layers):
        super(SupervisedDiffPool, self).__init__()

        # num_nodes = ceil(0.25 * in_channels)
        self.gnn1_pool = SageForDiffPool(
            in_channels, hidden_channels, 20, num_sage_layers
        )
        self.gnn1_embed = SageForDiffPool(
            in_channels, hidden_channels, hidden_channels, num_sage_layers
        )

        self.gnn2_embed = SageForDiffPool(
            hidden_channels, hidden_channels, hidden_channels, num_sage_layers
        )

        self.lin1 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, adjs):
        s = self.gnn1_pool(x, adjs)
        x = self.gnn1_embed(x, adjs)

        clusters = _get_argmax_cluster(s)

        # TODO: find a way to calculate the coarse adj matrix for bipartite graphs and
        # TODO: pass that through for more coarsening
        # x, coarse_adjs, ent_loss1 = _diff_pool_no_link_loss(x, adjs, s)
        # x = self.gnn2_embed(x, coarse_adjs)

        return F.log_softmax(x, dim=-1)


def _diff_pool_no_link_loss(x, adjs, s):
    # to_dense_adj(adjs[-1].edge_index)[0]
    # out_adj = torch.matmul(torch.matmul(s.transpose(1, 2), adj), s)

    s = torch.softmax(s, dim=-1)

    out = torch.matmul(s.transpose(0, 1), x)

    ent_loss = (-s * torch.log(s + 1e-15)).sum(dim=-1).mean()

    return out, adjs, ent_loss


def _get_argmax_cluster(s):
    s = torch.softmax(s, dim=-1)
    node_clusters = s.argmax(dim=-1)
    return node_clusters


class TopKPoolNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(TopKPoolNet, self).__init__()

        self.conv1 = GraphConv(in_channels, 128)
        self.pool1 = TopKPooling(128, ratio=0.8)
        self.conv2 = GraphConv(128, 128)
        self.pool2 = TopKPooling(128, ratio=0.8)
        self.conv3 = GraphConv(128, 128)
        self.pool3 = TopKPooling(128, ratio=num_classes)


    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = F.relu(self.conv1(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)

        x = F.relu(self.conv2(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)

        x = F.relu(self.conv3(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool3(x, edge_index, None, batch)

        x = F.log_softmax(x, dim=-1)

        return x


class TopKPoolNetNeighbour(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(TopKPoolNetNeighbour, self).__init__()

        self.conv1 = GraphConv(in_channels, 128)
        self.pool1 = TopKPooling(128, ratio=0.8)
        self.conv2 = GraphConv(128, 128)
        self.pool2 = TopKPooling(128, ratio=0.8)
        self.conv3 = GraphConv(128, 128)
        self.pool3 = TopKPooling(128, ratio=num_classes)


    def forward(self, x, adjs):
        # TODO: will this work with the neighbour sampler?
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[: size[1]]  # Target nodes are always placed first.

            x = F.relu(self.conv1((x, x_target), edge_index))
            x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, 1)

            x = F.relu(self.conv2(x, edge_index))
            x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, 1)

            x = F.relu(self.conv3(x, edge_index))
            x, edge_index, _, batch, _, _ = self.pool3(x, edge_index, None, 1)

        x = F.log_softmax(x, dim=-1)

        return x
