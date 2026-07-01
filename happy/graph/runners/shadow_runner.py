import torch
from torch_geometric.loader import ShaDowKHopSampler

from happy.models.shadow import ShaDowGCN
from happy.graph.runners.base import TrainRunner, BatchResult, _default_criterion


class ShaDowRunner(TrainRunner):
    '''
    Uses ShaDowKHopSampler — extracts localised k-hop subgraphs around each node. 
    Both train and val use the same sampler (val samples all nodes). 
    process_batch passes batch.batch and batch.root_n_id to the model. 
    validate accumulates batched outputs and concatenates them rather than calling model.inference.
    '''
    def setup_dataloader(self):
        train_loader = ShaDowKHopSampler(
            self.params.data,
            depth=6,
            num_neighbors=self.params.num_neighbours,
            node_idx=self.params.data.train_mask,
            batch_size=self.params.batch_size,
            num_workers=self.params.num_workers,
        )
        val_loader = ShaDowKHopSampler(
            self.params.data,
            depth=6,
            num_neighbors=self.params.num_neighbours,
            node_idx=None,
            batch_size=self.params.batch_size,
            num_workers=self.params.num_workers,
            shuffle=False,
        )
        return train_loader, val_loader

    def setup_model(self):
        return ShaDowGCN(
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
        )

    def process_batch(self, batch) -> BatchResult:
        out = self.model(batch.x, batch.edge_index, batch.batch, batch.root_n_id)
        loss = self.criterion(out, batch.y)
        loss.backward()
        self.optimiser.step()
        nodes = out.size()[0]
        return BatchResult(
            loss=loss,
            correct_predictions=int(out.argmax(dim=-1).eq(batch.y).sum().item()),
            nodes=nodes,
        )

    @torch.no_grad()
    def validate(self):
        data = self.params.data
        self.model.eval()
        out = []
        for batch in self.val_loader:
            batch = batch.to(self.params.device)
            batch_out = self.model(
                batch.x, batch.edge_index, batch.batch, batch.root_n_id
            )
            out.append(batch_out)
        out = torch.cat(out, dim=0).argmax(dim=-1)
        y = data.y.to(out.device)
        train_accuracy = int((out[data.train_mask].eq(y[data.train_mask])).sum()) / int(
            data.train_mask.sum()
        )
        val_accuracy = int((out[data.val_mask].eq(y[data.val_mask])).sum()) / int(
            data.val_mask.sum()
        )
        return train_accuracy, val_accuracy
