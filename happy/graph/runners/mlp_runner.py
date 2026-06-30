import torch
from torch_geometric.loader import DataLoader

from happy.models.mlp import MLP
from happy.graph.runners.base import TrainRunner, BatchResult, _default_criterion


class MLPRunner(TrainRunner):
    '''
    Like SIGN, uses plain index DataLoaders — the MLP ignores graph structure entirely, 
    treating it as a node classification problem on features only. 
    validate accumulates batched inference outputs the same way as SIGNRunner.
    '''
    def setup_dataloader(self):
        train_idx = self.params.data.train_mask.nonzero(as_tuple=False).view(-1)
        val_idx = self.params.data.val_mask.nonzero(as_tuple=False).view(-1)
        train_loader = DataLoader(
            train_idx,
            batch_size=self.params.batch_size,
            shuffle=True,
            num_workers=self.params.num_workers,
        )
        val_loader = DataLoader(
            val_idx, batch_size=self.params.batch_size, shuffle=False
        )
        return train_loader, val_loader

    def setup_model(self):
        return MLP(
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
        data = self.params.data
        train_x = data.x[batch].to(self.params.device)
        train_y = data.y[batch].to(self.params.device)
        out = self.model(train_x)
        loss = self.criterion(out, train_y)
        loss.backward()
        self.optimiser.step()
        nodes = data.train_mask[batch].sum().item()
        return BatchResult(
            loss=loss,
            correct_predictions=int((out.argmax(dim=-1).eq(train_y)).sum()),
            nodes=nodes,
        )

    @torch.no_grad()
    def validate(self):
        data = self.params.data
        self.model.eval()
        train_out = []
        for idx in self.train_loader:
            eval_x = data.x[idx].to(self.params.device)
            out_i, _ = self.model.inference(eval_x)
            train_out.append(out_i)
        train_out = torch.cat(train_out, dim=0).argmax(dim=-1)
        y = data.y.to(train_out.device)
        train_accuracy = int((train_out.eq(y[data.train_mask])).sum()) / int(
            data.train_mask.sum()
        )
        val_out = []
        for idx in self.val_loader:
            eval_x = data.x[idx].to(self.params.device)
            out_i, _ = self.model.inference(eval_x)
            val_out.append(out_i)
        val_out = torch.cat(val_out, dim=0).argmax(dim=-1)
        val_accuracy = int((val_out.eq(y[data.val_mask])).sum()) / int(
            data.val_mask.sum()
        )
        return train_accuracy, val_accuracy
