from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn as nn
from torch import Tensor

from torch_geometric.nn.conv import SAGEConv
from torch_geometric.nn.pool import TopKPooling, knn_graph
from torch_geometric.transforms import Distance
from torch_geometric.data import Data
from torch_geometric.nn.aggr import Aggregation
from torch_geometric.typing import Adj, OptPairTensor, Size, SparseTensor
from torch_geometric.nn import MessagePassing, GINConv, knn
from torch_geometric.utils import spmm, subgraph


class WeightedSAGEConv(SAGEConv):
    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        aggr: Optional[Union[str, List[str], Aggregation]] = "mean",
        normalize: bool = False,
        root_weight: bool = True,
        project: bool = False,
        bias: bool = True,
        **kwargs,
    ):
        super().__init__(
            in_channels,
            out_channels,
            aggr,
            normalize,
            root_weight,
            project,
            bias,
            **kwargs,
        )

    def forward(
        self,
        x: Union[Tensor, OptPairTensor],
        edge_index: Adj,
        edge_weight: Tensor = None,
        size: Size = None,
    ) -> Tensor:
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        if self.project and hasattr(self, "lin"):
            x = (self.lin(x[0]).relu(), x[1])

        out = self.propagate(edge_index, x=x, edge_weight=edge_weight, size=size)
        out = self.lin_l(out)

        x_r = x[1]
        if self.root_weight and x_r is not None:
            out = out + self.lin_r(x_r)

        if self.normalize:
            out = F.normalize(out, p=2.0, dim=-1)

        return out

    def message(self, x_j: Tensor, edge_weight: Tensor) -> Tensor:
        return x_j if edge_weight is None else x_j * edge_weight.view(-1, 1)

    def message_and_aggregate(
        self, adj_t: SparseTensor, x: OptPairTensor, edge_weight: Tensor = None
    ) -> Tensor:
        if isinstance(adj_t, SparseTensor):
            if edge_weight is not None:
                adj_t = adj_t.set_value(edge_weight, layout=None)
            else:
                adj_t = adj_t.set_value(None, layout=None)
        return spmm(adj_t, x[0], reduce=self.aggr)


class TopKPoolKnnEdges(TopKPooling):
    def __init__(self, in_channels, ratio, **kwargs):
        super().__init__(in_channels, ratio, **kwargs)

    def forward(self, x, edge_index, pos=None, edge_attr=None, batch=None, attn=None):
        x, edge_index, edge_attr, batch, perm, score = super().forward(
            x, edge_index, edge_attr, batch, attn
        )
        pos, edge_index, edge_attr = self._reconstruct_edges(
            pos, edge_attr, batch, perm
        )
        return x, pos, edge_index, edge_attr, batch, perm, score

    def _reconstruct_edges(self, pos, edge_attr, batch, perm):
        pos = pos[perm]
        edge_index = knn_graph(pos, k=6, batch=batch, loop=True)
        temp_data = Data(pos=pos, edge_index=edge_index, edge_attr=edge_attr)
        temp_data = Distance(cat=False, norm=True)(temp_data)
        edge_attr = temp_data.edge_attr
        return pos, edge_index, edge_attr


class KnnEdges(nn.Module):
    def __init__(self, start_k=6, k_increment=0, no_op=False):
        super().__init__()
        self.start_k = start_k
        self.k_increment = k_increment
        self.no_op = no_op

    def forward(self, x, pos, edge_index, edge_weight, batch, perm, score, i):
        if self.no_op:
            return x, pos, edge_index, edge_weight, batch, perm, score
        k = self.start_k + (self.k_increment * i)
        pos = pos[perm]
        edge_index = knn_graph(pos, k=k, batch=batch, loop=True)
        if edge_weight is not None:
            temp_data = Data(pos=pos, edge_index=edge_index)
            temp_data = Distance(cat=False, norm=True)(temp_data)
            edge_weight = temp_data.edge_attr[:, 0]
        return x, pos, edge_index, edge_weight, batch, perm, score


def pool_one_hop(edge_index, num_nodes, iteration_size, reduction_factor=0.75, i=0):
    # Reduce the iteration_size by reduction_factor for each iteration i
    iteration_size = int(iteration_size * (reduction_factor ** i))

    device = edge_index.device
    perm = []  # This will store our super nodes
    node_mask = torch.ones(num_nodes, dtype=torch.bool, device=device)

    while node_mask.any():  # While there are nodes left
        # Filter edge_index for available nodes
        available_edge_mask = node_mask[edge_index[0]] & node_mask[edge_index[1]]
        filtered_edge_index = edge_index[:, available_edge_mask]

        # Randomly select iteration_size nodes which are still available
        available_nodes = torch.where(node_mask)[0]
        if int(available_nodes.size(0) * reduction_factor) >= iteration_size:
            supernodes = available_nodes[
                torch.randint(0, available_nodes.size(0), (iteration_size,))
            ]
        else:
            # Reduce the iteration size if we can't select enough nodes
            iteration_size = int(reduction_factor * available_nodes.size(0))
            if iteration_size <= 1:
                break
            continue

        # Remove the neighbors of supernodes from the selection pool
        rows, cols = filtered_edge_index

        # Step 1: Create a mask of supernode-to-supernode relationships
        supernode_to_supernode_mask = (
            torch.isin(rows, supernodes) & torch.isin(cols, supernodes) & (rows != cols)
        )
        # Step 2: Identify supernodes that have direct neighbors among supernodes
        direct_neighbors = torch.cat(
            (rows[supernode_to_supernode_mask], cols[supernode_to_supernode_mask])
        )
        to_remove = torch.unique(direct_neighbors)
        # Update supernodes by removing those that have direct neighbors among supernodes
        supernodes = supernodes[~torch.isin(supernodes, to_remove)]

        if supernodes.size(0) == 0:
            iteration_size = int(iteration_size * reduction_factor)
            if iteration_size <= 1:
                break
            continue

        # Append supernodes to perm
        perm.extend(supernodes.tolist())

        # Exclude supernodes from the neighbors list to account for self-loops
        neighbors = torch.unique(cols[torch.isin(rows, supernodes)])
        neighbors = neighbors[~torch.isin(neighbors, supernodes)]

        node_mask[supernodes] = False
        node_mask[neighbors] = False

    return torch.tensor(perm, dtype=torch.long)


def pool_one_hop_old(edge_index, num_nodes, iteration_size):
    device = edge_index.device
    perm = []  # This will store our super nodes
    node_mask = torch.ones(num_nodes, dtype=torch.bool, device=device)

    while node_mask.any():  # While there are nodes left
        # Randomly select iteration_size nodes which are still available
        available_nodes = torch.where(node_mask)[0]
        if available_nodes.size(0) > iteration_size:
            supernodes = available_nodes[
                torch.randint(0, available_nodes.size(0), (iteration_size,))
            ]
        else:
            supernodes = available_nodes

        # Append supernodes to perm
        perm.extend(supernodes.tolist())

        # Remove the neighbors of supernodes from the selection pool
        rows, cols = edge_index
        neighbors = torch.unique(cols[torch.isin(rows, supernodes)])
        # Exclude supernodes from the neighbors list to account for self-loops
        neighbors = neighbors[~torch.isin(neighbors, supernodes)]

        node_mask[supernodes] = False
        node_mask[neighbors] = False

    return torch.tensor(perm, dtype=torch.long)


def pool_subgraph(pos, pool_ratio):
    # Randomly select super nodes.
    num_to_keep = int(pos.shape[0] * pool_ratio)
    super_node_indices = torch.randperm(pos.shape[0], device=pos.device)[:num_to_keep]

    # Assign nodes to the nearest supernode based on their positions.
    nearest_super_node = knn(pos[super_node_indices], pos, k=1)[1]
    print("knn complete")

    from happy.graph.utils.visualise_points import visualize_points
    from happy.organs import get_organ

    organ = get_organ("placenta")
    pos = pos.to("cpu").numpy()
    visualize_points(
        organ,
        "nearest_super_node.png",
        pos,
        colours=nearest_super_node.to("cpu").numpy(),
        width=int(pos[:, 0].max()) - int(pos[:, 0].min()),
        height=int(pos[:, 1].max()) - int(pos[:, 1].min()),
    )


class SubgraphPool(MessagePassing):
    def __init__(self, in_channels, out_channels, pool_ratio, **kwargs):
        super(SubgraphPool, self).__init__(aggr="add", **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pool_ratio = pool_ratio

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(in_channels, 2 * in_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(2 * in_channels, out_channels),
        )
        self.conv = GINConv(self.mlp)

    def forward(self, x, edge_index, pos):
        # Randomly select super nodes
        num_to_keep = int(pos.shape[0] * self.pool_ratio)
        supernode_indices = torch.randperm(pos.shape[0])[:num_to_keep]

        # Assign remaining nodes to the nearest supernode based on their positions
        nearest_supernode = knn(pos[supernode_indices], pos, k=1)[1]

        pooled_features = []
        # Iterate through each super node
        for idx in supernode_indices:
            # Get nodes that have been assigned to this super node
            child_nodes = torch.where(nearest_supernode.eq(idx))[0]

            # if no child nodes, continue to the next super node
            if child_nodes.size(0) == 0:
                continue

            # Extract child subgraph and apply GINConv on the subgraph
            child_edge_index = subgraph(child_nodes, edge_index)
            child_x = self.conv(x[child_nodes], child_edge_index)

            # Extract the feature of the super node
            super_node_feature = child_x[x[child_nodes] == idx]
            pooled_features.append(super_node_feature)

        pooled_features = torch.cat(pooled_features, dim=0)
        return pooled_features
