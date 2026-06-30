import torch
from torch_geometric.loader import GraphSAINTRandomWalkSampler, NeighborLoader
from torch_geometric.utils import degree

from happy.models.graphsaint import GraphSAINT
from happy.graph.runners.base import TrainRunner, BatchResult, _default_criterion


class GraphSAINTRunner(TrainRunner):
    '''
    Uses GraphSAINTRandomWalkSampler — random walk subgraph sampling with importance-weighted loss (node_norm * edge_norm).
    Overrides prepare_data to precompute edge weights from node degree. 
    Overrides validate to switch the model aggregation from "add" (training) to "mean" (inference).
    '''
    def prepare_data(self):
        row, col = self.params.data.edge_index
        self.params.data.edge_weight = (
            1.0 / degree(col, self.params.data.num_nodes)[col]
        )
        super().prepare_data()

    def setup_dataloader(self):
        train_loader = GraphSAINTRandomWalkSampler(
            self.params.data,
            batch_size=self.params.batch_size,
            walk_length=self.params.layers,
            num_steps=30,
            sample_coverage=self.params.num_neighbours,
            shuffle=True,
            num_workers=self.params.num_workers,
        )
        val_loader = NeighborLoader(
            self.params.data,
            num_neighbors=[-1],
            batch_size=1024,
            shuffle=False,
            num_workers=self.params.num_workers,
        )
        val_loader.data.num_nodes = self.params.data.num_nodes
        val_loader.data.n_id = torch.arange(self.params.data.num_nodes)
        return train_loader, val_loader

    def setup_model(self):
        return GraphSAINT(
            self.params.data.num_node_features,
            hidden_channels=self.params.hidden_units,
            out_channels=self.num_classes,
            num_layers=self.params.layers,
            dropout=self.params.dropout,
        )

    def setup_criterion(self):
        return _default_criterion(
            self.params.weighted_loss,
            self.params.data,
            self.params.organ,
            self.params.custom_weights,
            self.params.device,
            reduction="none",
        )

    def process_batch(self, batch) -> BatchResult:
        self.model.set_aggr("add")
        edge_weight = batch.edge_norm * batch.edge_weight
        out = self.model(batch.x, batch.edge_index, edge_weight)
        loss = self.criterion(out, batch.y)
        loss = (loss * batch.node_norm)[batch.train_mask].sum()
        loss.backward()
        self.optimiser.step()
        nodes = batch.train_mask.sum().item()
        return BatchResult(
            loss=loss,
            correct_predictions=int(
                out[batch.train_mask]
                .argmax(dim=-1)
                .eq(batch.y[batch.train_mask])
                .sum()
                .item()
            ),
            nodes=nodes,
        )

    @torch.no_grad()
    def validate(self):
        data = self.params.data
        self.model.eval()
        self.model.set_aggr("mean")
        out, _ = self.model.inference(data.x, self.val_loader, self.params.device)
        out = out.argmax(dim=-1)
        y = data.y.to(out.device)
        train_accuracy = int((out[data.train_mask].eq(y[data.train_mask])).sum()) / int(
            data.train_mask.sum()
        )
        val_accuracy = int((out[data.val_mask].eq(y[data.val_mask])).sum()) / int(
            data.val_mask.sum()
        )
        return train_accuracy, val_accuracy
