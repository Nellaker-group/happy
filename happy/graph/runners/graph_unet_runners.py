import torch

from happy.models.graph_unet import GraphUNet, GraphUNetNoSkip
from happy.graph.runners.base import TrainRunner, _default_criterion


class GraphUNetRunner(TrainRunner):
    '''
    GraphUNet operates on the full graph at once (no mini-batching). 
    train() and validate() are fully overridden — no loader loop, just a single forward pass on data.x and data.edge_index. 
    Returns (loss, accuracy) directly rather than going through BatchResult. 
    GraphUNetNoSkipRunner just swaps the model variant.
    '''
    def setup_model(self):
        return GraphUNet(
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

    def train(self):
        self.model.train()
        self.optimiser.zero_grad()
        out = self.model(
            self.params.data.x.to(self.params.device),
            self.params.data.edge_index.to(self.params.device),
        )
        train_idx = self.params.data.train_mask.nonzero(as_tuple=False).view(-1)
        train_out = out[train_idx]
        train_y = self.params.data.y[train_idx].to(self.params.device)
        loss = self.criterion(train_out, train_y)
        loss.backward()
        self.optimiser.step()
        acc = int(train_out.argmax(dim=-1).eq(train_y).sum().item()) / int(train_y.size()[0])
        return loss.item(), acc

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        out = self.model(
            self.params.data.x.to(self.params.device),
            self.params.data.edge_index.to(self.params.device),
        )
        val_idx = self.params.data.val_mask.nonzero(as_tuple=False).view(-1)
        val_out = out[val_idx]
        val_y = self.params.data.y[val_idx].to(self.params.device)
        loss = self.criterion(val_out, val_y)
        acc = int(val_out.argmax(dim=-1).eq(val_y).sum().item()) / int(val_y.size()[0])
        return loss.item(), acc


class GraphUNetNoSkipRunner(GraphUNetRunner):
    def setup_model(self):
        return GraphUNetNoSkip(
            self.params.data.num_node_features,
            hidden_channels=self.params.hidden_units,
            out_channels=self.num_classes,
            num_layers=self.params.layers,
            dropout=self.params.dropout,
        )
