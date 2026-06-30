import copy

import torch
from torch_geometric.loader import NeighborLoader

from happy.models.graphsage import SupervisedSAGE
from happy.graph.runners.base import TrainRunner, BatchResult, _compute_tissue_weights


class GraphSAGERunner(TrainRunner):
    '''
    Uses NeighborLoader — samples a fixed number of neighbours per layer for each seed node in the training mask. 
    The val loader uses all neighbours ([-1]) over the whole graph. 
    Uses CrossEntropyLoss (SAGE model outputs logits, not log-probs). 
    process_batch only trains on nodes within batch.batch_size that are in the training mask.
    '''
    def setup_dataloader(self):
        train_loader = NeighborLoader(
            self.params.data,
            input_nodes=self.params.data.train_mask,
            num_neighbors=[
                self.params.num_neighbours for _ in range(self.params.layers)
            ],
            batch_size=self.params.batch_size,
            shuffle=True,
            num_workers=self.params.num_workers,
        )
        val_loader = NeighborLoader(
            copy.copy(self.params.data),
            num_neighbors=[-1],
            shuffle=False,
            batch_size=512,
        )
        val_loader.data.num_nodes = self.params.data.num_nodes
        val_loader.data.n_id = torch.arange(self.params.data.num_nodes)
        return train_loader, val_loader

    def setup_model(self):
        return SupervisedSAGE(
            self.params.data.num_node_features,
            hidden_channels=self.params.hidden_units,
            out_channels=self.num_classes,
            num_layers=self.params.layers,
            dropout=self.params.dropout,
        )

    def setup_criterion(self):
        if self.params.weighted_loss:
            data_classes = self.params.data.y[self.params.data.train_mask].numpy()
            class_weights = _compute_tissue_weights(
                data_classes, self.params.organ, self.params.custom_weights
            )
            class_weights = torch.FloatTensor(class_weights)
            class_weights = class_weights.to(self.params.device)
            return torch.nn.CrossEntropyLoss(weight=class_weights)
        return torch.nn.CrossEntropyLoss()

    # todo: do we need [:batch.batch_size] here?
    def process_batch(self, batch) -> BatchResult:
        batch_train_nodes = batch.train_mask[: batch.batch_size]
        batch_y = batch.y[: batch.batch_size]

        out = self.model(batch.x, batch.edge_index)[: batch.batch_size]
        train_out = out[batch_train_nodes]
        train_y = batch_y[batch_train_nodes]
        loss = self.criterion(train_out, train_y)
        loss.backward()
        self.optimiser.step()

        nodes = batch_train_nodes.sum().item()
        return BatchResult(
            loss=loss,
            correct_predictions=int(train_out.argmax(dim=-1).eq(train_y).sum().item()),
            nodes=nodes,
        )
