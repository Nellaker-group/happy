import json
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.utils import dropout_node
from sklearn.utils.class_weight import compute_class_weight

from happy.organs import Organ
from happy.graph.enums import SupervisedModelsArg


@dataclass
class TrainParams:
    data: Data
    device: str
    pretrained: Optional[str]
    model_type: SupervisedModelsArg
    batch_size: int
    num_neighbours: int
    epochs: int
    layers: int
    hidden_units: int
    dropout: float
    node_dropout: float
    learning_rate: float
    num_workers: int
    weighted_loss: bool
    custom_weights: bool
    validation_step: int
    organ: Organ

    def save(self, seed, exp_name, run_path):
        to_save = {k: v for k, v in asdict(self).items() if k not in ("data", "organ")}
        to_save["seed"] = seed
        to_save["exp_name"] = exp_name
        with open(run_path / "train_params.json", "w") as f:
            json.dump(to_save, f, indent=2)


@dataclass
class BatchResult:
    loss: torch.Tensor
    correct_predictions: int
    nodes: int


@dataclass
class BatchResultMultilabel:
    loss: torch.Tensor
    correct_predictions_1: int
    correct_predictions_2: int
    nodes: int


class TrainRunner:
    def __init__(self, params: TrainParams):
        self.params: TrainParams = params
        self._model: Optional[nn.Module] = None
        self._train_loader = None
        self._val_loader = None
        self._optimiser: Optional[torch.optim.Optimizer] = None
        self._criterion: Optional[nn.Module] = None
        self.num_classes = len(self.params.organ.tissues)

    @staticmethod
    def new(params: TrainParams) -> "TrainRunner":
        from happy.graph.runners.graphsage_runner import GraphSAGERunner
        from happy.graph.runners.clustergcn_runners import (
            ClusterGCNRunner,
            ClusterGCNJumpingRunner,
            ClusterGCNEdgeRunner,
            ClusterGINRunner,
            ClusterGINEdgeRunner,
            GATRunner,
            GATV2Runner,
            ClusterGCNMultilabelRunner,
            ClusterGCNMultiheadRunner,
        )
        from happy.graph.runners.graphsaint_runner import GraphSAINTRunner
        from happy.graph.runners.shadow_runner import ShaDowRunner
        from happy.graph.runners.sign_runner import SIGNRunner
        from happy.graph.runners.graph_unet_runners import GraphUNetRunner, GraphUNetNoSkipRunner
        from happy.graph.runners.mlp_runner import MLPRunner

        cls = {
            SupervisedModelsArg.sup_graphsage: GraphSAGERunner,
            SupervisedModelsArg.sup_clustergcn: ClusterGCNRunner,
            SupervisedModelsArg.sup_jumping: ClusterGCNJumpingRunner,
            SupervisedModelsArg.sup_clustergcne: ClusterGCNEdgeRunner,
            SupervisedModelsArg.sup_clustergin: ClusterGINRunner,
            SupervisedModelsArg.sup_clustergine: ClusterGINEdgeRunner,
            SupervisedModelsArg.sup_gat: GATRunner,
            SupervisedModelsArg.sup_gatv2: GATV2Runner,
            SupervisedModelsArg.sup_graphsaint: GraphSAINTRunner,
            SupervisedModelsArg.sup_shadow: ShaDowRunner,
            SupervisedModelsArg.sup_sign: SIGNRunner,
            SupervisedModelsArg.sup_mlp: MLPRunner,
            SupervisedModelsArg.sup_graphunet: GraphUNetRunner,
            SupervisedModelsArg.sup_graphunet_noskip: GraphUNetNoSkipRunner,
            SupervisedModelsArg.sup_clustergcn_multilabel: ClusterGCNMultilabelRunner,
            SupervisedModelsArg.sup_clustergcn_multihead: ClusterGCNMultiheadRunner,
        }
        ModelClass = cls[params.model_type]
        return ModelClass(params)

    @property
    def model(self):
        if self._model is None:
            if self.params.pretrained is not None:
                self._model = torch.load(self.params.pretrained)
            else:
                self._model = self.setup_model()
            self._model = self._model.to(self.params.device)
        return self._model

    @property
    def train_loader(self):
        if self._train_loader is None:
            self._setup_loaders()
        return self._train_loader

    @property
    def val_loader(self):
        if self._val_loader is None:
            self._setup_loaders()
        return self._val_loader

    @property
    def optimiser(self):
        if self._optimiser is None:
            self._setup_optimiser()
        return self._optimiser

    @property
    def criterion(self):
        if self._criterion is None:
            self._criterion = self.setup_criterion()
        return self._criterion

    def _setup_loaders(self):
        self._train_loader, self._val_loader = self.setup_dataloader()

    def _setup_optimiser(self):
        self._optimiser = torch.optim.Adam(
            self.model.parameters(), lr=self.params.learning_rate
        )

    def prepare_data(self):
        self.params.data.x.to(self.params.device)
        self.params.data.edge_index.to(self.params.device)

    @classmethod
    def setup_dataloader(cls):
        raise NotImplementedError(f"setup_dataloader not implemented for {cls.__name__}")

    @classmethod
    def setup_model(cls):
        raise NotImplementedError(f"setup_model not implemented for {cls.__name__}")

    @classmethod
    def setup_criterion(cls):
        raise NotImplementedError(f"setup_criterion not implemented for {cls.__name__}")

    def train(self):
        self.model.train()
        total_loss = 0
        total_examples = 0
        total_correct = 0
        for batch in self.train_loader:
            batch = batch.to(self.params.device)
            self.optimiser.zero_grad()

            batch.edge_index, edge_mask, node_mask = dropout_node(
                batch.edge_index, p=self.params.node_dropout, num_nodes=batch.num_nodes
            )
            batch.edge_attr = batch.edge_attr[edge_mask, :]
            if hasattr(batch, "edge_weight"):
                if batch.edge_weight is not None:
                    batch.edge_weight = batch.edge_weight[edge_mask]
            if hasattr(batch, "edge_norm"):
                if batch.edge_norm is not None:
                    batch.edge_norm = batch.edge_norm[edge_mask]

            result = self.process_batch(batch)

            total_loss += result.loss.item() * result.nodes
            total_correct += result.correct_predictions
            total_examples += result.nodes
        return total_loss / total_examples, total_correct / total_examples

    @classmethod
    def process_batch(cls, batch) -> BatchResult:
        raise NotImplementedError(f"process_batch not implemented for {cls.__name__}")

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

    def save_state(self, run_path, epoch):
        torch.save(self.model, run_path / f"{epoch}_graph_model.pt")


def _default_criterion(weighted_loss, data, organ, custom_weights, device, reduction="mean"):
    if weighted_loss:
        data_classes = data.y[data.train_mask].numpy()
        class_weights = _compute_tissue_weights(data_classes, organ, custom_weights)
        class_weights = torch.FloatTensor(class_weights)
        class_weights = class_weights.to(device)
        criterion = torch.nn.NLLLoss(weight=class_weights, reduction=reduction)
    else:
        criterion = torch.nn.NLLLoss(reduction=reduction)
    return criterion


def _compute_tissue_weights(data_classes, organ, custom_weights):
    unique_classes = np.unique(data_classes)
    if not custom_weights:
        weighting = "balanced"
    else:
        custom_weights = [1, 0.85, 0.9, 5, 1, 1.3, 5.6, 3, 50]
        weighting = dict(zip(list(unique_classes), custom_weights))
    class_weights = compute_class_weight(
        weighting, classes=unique_classes, y=data_classes
    )
    # Account for missing tissues in training data
    classes_in_training = set(unique_classes)
    all_classes = {tissue.id for tissue in organ.tissues}
    missing_classes = list(all_classes - classes_in_training)
    missing_classes.sort()
    for i in missing_classes:
        class_weights = np.insert(class_weights, i, 0.0)
    return class_weights
