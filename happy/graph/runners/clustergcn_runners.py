import torch
import torch.nn.functional as F
import torch
from torch_geometric.loader import ClusterData, ClusterLoader, NeighborLoader
from torch_geometric.data import Data
from torch_geometric.utils import dropout_node

from happy.models.clustergcn import (
    ClusterGCN,
    JumpingClusterGCN,
    ClusterGCNEdges,
    ClusterGCNMultilabel,
    ClusterGCNMultihead,
)
from happy.models.gin import ClusterGIN
from happy.models.gat import GAT, GATv2
from happy.graph.runners.base import TrainRunner, BatchResult, BatchResultMultilabel, _default_criterion


class ClusterGCNRunner(TrainRunner):
    '''
    Shared ClusterLoader for training (partitions the graph into clusters, samples cluster subgraphs) 
    and NeighborSampler for validation.
    '''
    def setup_dataloader(self):
        # ClusterData and NeighborLoader both mishandle DataBatch: batch/ptr
        # attributes cause attribute slicing to use the wrong sizes
        flat_data = Data(**{
            k: v for k, v in self.params.data.items()
            if k not in ('batch', 'ptr', 'n_id')
        })
        flat_data.num_nodes = int(self.params.data.x.size(0))  # prevents wrong is_node_attr caching for multi-slide graphs
        # ClusterData only partitions attributes whose length == num_nodes. If `y`
        # is misaligned it would be left at full size and crash mid-epoch, so fail here with a clear message.
        if getattr(flat_data, "y", None) is not None and not flat_data.is_node_attr("y"):
            raise ValueError(
                f"Labels `y` ({flat_data.y.size(0)}) are not aligned with the graph "
                f"nodes ({flat_data.num_nodes}); ClusterData cannot partition them. "
                f"Check that every slide's labels match its node count."
            )
        cluster_data = ClusterData(
            flat_data,
            num_parts=int(flat_data.num_nodes / self.params.num_neighbours),
            recursive=False,
        )
        train_loader = ClusterLoader(
            cluster_data,
            batch_size=self.params.batch_size,
            shuffle=True,
            num_workers=self.params.num_workers,
        )
        val_loader = NeighborLoader(
            flat_data,
            num_neighbors=[-1],
            batch_size=1024,
            input_nodes=torch.arange(flat_data.num_nodes),
            shuffle=False,
            num_workers=self.params.num_workers,
        )
        return train_loader, val_loader

    def setup_model(self):
        return ClusterGCN(
            self.params.data.num_node_features,
            hidden_channels=self.params.hidden_units,
            out_channels=self.num_classes,
            num_layers=self.params.layers,
            dropout=self.params.dropout,
            reduce_dims=64,
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
        out = self.model(batch.x, batch.edge_index)
        train_out = out[batch.train_mask]
        train_y = batch.y[batch.train_mask]
        loss = self.criterion(train_out, train_y)
        loss.backward()
        self.optimiser.step()
        nodes = batch.train_mask.sum().item()
        return BatchResult(
            loss=loss,
            correct_predictions=int(train_out.argmax(dim=-1).eq(train_y).sum().item()),
            nodes=nodes,
        )


class ClusterGCNJumpingRunner(ClusterGCNRunner):
    def setup_model(self):
        return JumpingClusterGCN(
            self.params.data.num_node_features,
            hidden_channels=self.params.hidden_units,
            out_channels=self.num_classes,
            num_layers=self.params.layers,
            dropout=self.params.dropout,
        )


class ClusterGCNEdgeRunner(ClusterGCNRunner):
    def setup_model(self):
        return ClusterGCNEdges(
            self.params.data.num_node_features,
            hidden_channels=self.params.hidden_units,
            out_channels=self.num_classes,
            num_layers=self.params.layers,
            dropout=self.params.dropout,
            reduce_dims=64,
        )

    def process_batch(self, batch) -> BatchResult:
        out = self.model(batch.x, batch.edge_index, batch.edge_attr)
        train_out = out[batch.train_mask]
        train_y = batch.y[batch.train_mask]
        loss = self.criterion(train_out, train_y)
        loss.backward()
        self.optimiser.step()
        nodes = batch.train_mask.sum().item()
        return BatchResult(
            loss=loss,
            correct_predictions=int(train_out.argmax(dim=-1).eq(train_y).sum().item()),
            nodes=nodes,
        )

    @torch.no_grad()
    def validate(self):
        data = self.params.data
        self.model.eval()
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


class ClusterGINRunner(ClusterGCNEdgeRunner):
    def setup_model(self):
        return ClusterGIN(
            self.params.data.num_node_features,
            hidden_channels=self.params.hidden_units,
            out_channels=self.num_classes,
            num_layers=self.params.layers,
            dropout=self.params.dropout,
            reduce_dims=64,
            include_edge_attr=False,
        )


class ClusterGINEdgeRunner(ClusterGCNEdgeRunner):
    def setup_model(self):
        return ClusterGIN(
            self.params.data.num_node_features,
            hidden_channels=self.params.hidden_units,
            out_channels=self.num_classes,
            num_layers=self.params.layers,
            dropout=self.params.dropout,
            reduce_dims=64,
            include_edge_attr=True,
        )


class GATRunner(ClusterGCNRunner):
    def setup_model(self):
        return GAT(
            self.params.data.num_node_features,
            hidden_channels=self.params.hidden_units,
            out_channels=self.num_classes,
            heads=4,
            num_layers=self.params.layers,
            dropout=self.params.dropout,
        )


class GATV2Runner(ClusterGCNRunner):
    def setup_model(self):
        return GATv2(
            self.params.data.num_node_features,
            hidden_channels=self.params.hidden_units,
            out_channels=self.num_classes,
            heads=4,
            num_layers=self.params.layers,
            dropout=self.params.dropout,
        )


class ClusterGCNMultilabelRunner(ClusterGCNRunner):
    def setup_model(self):
        return ClusterGCNMultilabel(
            self.params.data.num_node_features,
            hidden_channels=self.params.hidden_units,
            out_channels=self.num_classes,
            num_layers=self.params.layers,
            dropout=self.params.dropout,
            reduce_dims=64,
        )

    def setup_criterion(self):
        assert self.params.weighted_loss is False, "Weighted loss not supported for multilabel"
        return torch.nn.CrossEntropyLoss()

    def train(self):
        self.model.train()
        total_loss = 0
        total_examples = 0
        total_correct_1 = 0
        total_correct_2 = 0

        for batch in self.train_loader:
            batch = batch.to(self.params.device)
            self.optimiser.zero_grad()

            batch.edge_index, edge_mask, node_mask = dropout_node(
                batch.edge_index, p=self.params.node_dropout, num_nodes=batch.num_nodes
            )
            batch.edge_attr = batch.edge_attr[edge_mask, :]

            result = self.process_batch(batch)

            total_loss += result.loss.item() * result.nodes
            total_correct_1 += result.correct_predictions_1
            total_correct_2 += result.correct_predictions_2
            total_examples += result.nodes
        return total_loss / total_examples, total_correct_1 / total_examples, total_correct_2 / total_examples

    def process_batch(self, batch) -> BatchResultMultilabel:
        out = self.model(batch.x, batch.edge_index)
        train_out = out[batch.train_mask]
        train_y = batch.y[batch.train_mask]
        train_y_1 = train_y[:, 0]
        train_y_2 = train_y[:, 1]
        train_y_mhot = self.multihot_label(train_y)
        loss = self.criterion(train_out, train_y_mhot)
        loss.backward()
        self.optimiser.step()
        nodes = batch.train_mask.sum().item()
        train_pred_1 = train_out[:, :-2].argmax(dim=-1)
        train_pred_2 = train_out[:, -2:].argmax(dim=-1)
        train_pred_2 = train_pred_2 * -9 + 9
        return BatchResultMultilabel(
            loss=loss,
            correct_predictions_1=int(train_pred_1.eq(train_y_1).sum().item()),
            correct_predictions_2=int(train_pred_2.eq(train_y_2).sum().item()),
            nodes=nodes,
        )

    @torch.no_grad()
    def validate(self):
        data = self.params.data
        self.model.eval()
        out, _ = self.model.inference(data.x, self.val_loader, self.params.device)
        train_out = out[data.train_mask]
        train_pred_1 = train_out[:, :-2].argmax(dim=-1)
        train_pred_2 = (train_out[:, -2:].argmax(dim=-1)) * -9 + 9
        val_out = out[data.val_mask]
        val_pred_1 = val_out[:, :-2].argmax(dim=-1)
        val_pred_2 = (val_out[:, -2:].argmax(dim=-1)) * -9 + 9
        y = data.y.to(out.device)
        train_y = y[data.train_mask]
        val_y = y[data.val_mask]
        num_train = int(data.train_mask.sum())
        num_val = int(data.val_mask.sum())
        train_accuracy_1 = int((train_pred_1.eq(train_y[:, 0])).sum()) / num_train
        val_accuracy_1 = int((val_pred_1.eq(val_y[:, 0])).sum()) / num_val
        train_accuracy_2 = int((train_pred_2.eq(train_y[:, 1])).sum()) / num_train
        val_accuracy_2 = int((val_pred_2.eq(val_y[:, 1])).sum()) / num_val
        return train_accuracy_1, val_accuracy_1, train_accuracy_2, val_accuracy_2

    def multihot_label(self, y):
        y1 = y[:, 0]
        y2 = y[:, 1]
        y1 = torch.nn.functional.one_hot(y1, num_classes=self.num_classes)
        y2 = torch.nn.functional.one_hot(y2, num_classes=self.num_classes)
        y2[:, 0] = 0
        y_multi = y1 | y2
        y_multi[:, -1] = 1 - y_multi[:, -2]
        y_multi = y_multi.type(torch.FloatTensor)
        return y_multi.to(self.params.device)


class ClusterGCNMultiheadRunner(ClusterGCNMultilabelRunner):
    def setup_model(self):
        return ClusterGCNMultihead(
            self.params.data.num_node_features,
            hidden_channels=self.params.hidden_units,
            out_channels=self.num_classes,
            num_layers=self.params.layers,
            dropout=self.params.dropout,
            reduce_dims=64,
        )

    def setup_criterion(self):
        return torch.nn.NLLLoss()

    def process_batch(self, batch) -> BatchResultMultilabel:
        out_1, out_2 = self.model(batch.x, batch.edge_index)
        train_out_1 = out_1[batch.train_mask]
        train_out_2 = out_2[batch.train_mask]
        train_y = batch.y[batch.train_mask]
        train_y_1 = train_y[:, 0]
        train_y_2 = ((train_y[:, 1] - 9) / -9).type(torch.LongTensor).to(self.params.device)
        loss = self.criterion(train_out_1, train_y_1) + self.criterion(train_out_2, train_y_2)
        loss.backward()
        self.optimiser.step()
        nodes = batch.train_mask.sum().item()
        return BatchResultMultilabel(
            loss=loss,
            correct_predictions_1=int(train_out_1.argmax(dim=-1).eq(train_y_1).sum().item()),
            correct_predictions_2=int(train_out_2.argmax(dim=-1).eq(train_y_2).sum().item()),
            nodes=nodes,
        )

    @torch.no_grad()
    def validate(self):
        data = self.params.data
        self.model.eval()
        out, _ = self.model.inference(data.x, self.val_loader, self.params.device)
        train_out = out[data.train_mask]
        train_pred_1 = train_out[:, :-2].argmax(dim=-1)
        train_pred_2 = (train_out[:, -2:].argmax(dim=-1)) * -9 + 9
        val_out = out[data.val_mask]
        val_pred_1 = val_out[:, :-2].argmax(dim=-1)
        val_pred_2 = (val_out[:, -2:].argmax(dim=-1)) * -9 + 9
        y = data.y.to(out.device)
        train_y = y[data.train_mask]
        val_y = y[data.val_mask]
        num_train = int(data.train_mask.sum())
        num_val = int(data.val_mask.sum())
        train_accuracy_1 = int((train_pred_1.eq(train_y[:, 0])).sum()) / num_train
        val_accuracy_1 = int((val_pred_1.eq(val_y[:, 0])).sum()) / num_val
        train_accuracy_2 = int((train_pred_2.eq(train_y[:, 1])).sum()) / num_train
        val_accuracy_2 = int((val_pred_2.eq(val_y[:, 1])).sum()) / num_val
        return train_accuracy_1, val_accuracy_1, train_accuracy_2, val_accuracy_2
