import torch
from torch_geometric.transforms import SIGN
from torch_geometric.loader import DataLoader

from happy.models.sign import SIGN as SIGN_MLP
from happy.graph.runners.base import TrainRunner, BatchResult, _default_criterion


class SIGNRunner(TrainRunner):
    '''
    SIGN pre-computes diffused feature matrices (x, x1, x2, ...) before training rather than 
    sampling subgraphs at runtime. Overrides prepare_data to apply the SIGN transform. 
    Both loaders are plain index DataLoaders (just node indices, no graph structure). 
    process_batch and validate look up data[f"x{i}"] for each diffusion hop and pass the list to the model.
    '''
    def prepare_data(self):
        self.params.data = SIGN(self.params.layers)(self.params.data)
        super().prepare_data()

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
        return SIGN_MLP(
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
        sign_k = self.model.num_layers
        data = self.params.data
        train_x = [data.x[batch].to(self.params.device)]
        train_y = data.y[batch].to(self.params.device)
        train_x += [
            data[f"x{i}"][batch].to(self.params.device) for i in range(1, sign_k + 1)
        ]
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
            eval_x = [data.x[idx].to(self.params.device)]
            eval_x += [
                data[f"x{i}"][idx].to(self.params.device)
                for i in range(1, self.model.num_layers + 1)
            ]
            out_i, _ = self.model.inference(eval_x)
            train_out.append(out_i)
        train_out = torch.cat(train_out, dim=0).argmax(dim=-1)
        y = data.y.to(train_out.device)
        train_accuracy = int((train_out.eq(y[data.train_mask])).sum()) / int(
            data.train_mask.sum()
        )
        val_out = []
        for idx in self.val_loader:
            eval_x = [data.x[idx].to(self.params.device)]
            eval_x += [
                data[f"x{i}"][idx].to(self.params.device)
                for i in range(1, self.model.num_layers + 1)
            ]
            out_i, _ = self.model.inference(eval_x)
            val_out.append(out_i)
        val_out = torch.cat(val_out, dim=0).argmax(dim=-1)
        val_accuracy = int((val_out.eq(y[data.val_mask])).sum()) / int(
            data.val_mask.sum()
        )
        return train_accuracy, val_accuracy
