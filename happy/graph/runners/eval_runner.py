from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch_geometric.data import Data

# Attributes added to MessagePassing/nn.Module in newer PyG/PyTorch versions
# that are absent from models saved with older versions.
_COMPAT_ATTR_DEFAULTS = {
    "_lazy_load_hook": None,
    "_is_full_backward_hook": None,
    "_forward_pre_hooks_with_kwargs": {},
    "_forward_hooks_with_kwargs": {},
    "_forward_hooks_always_called": {},
    "_state_dict_pre_hooks": {},
    "_decomposed_layers": 1,
}

def _patch_module_getattr():
    _original = nn.Module.__getattr__
    def _patched(self, name: str):
        if name in _COMPAT_ATTR_DEFAULTS:
            default = _COMPAT_ATTR_DEFAULTS[name]
            object.__setattr__(self, name, default)
            return default
        return _original(self, name)
    nn.Module.__getattr__ = _patched
from torch_geometric.transforms import SIGN
from torch_geometric.loader import (
    DataLoader,
    NeighborLoader,
    ShaDowKHopSampler,
)

from happy.models.graphsaint import GraphSAINT
from happy.models.sign import SIGN as SIGN_MLP
from happy.organs import Organ
from happy.graph.enums import SupervisedModelsArg


@dataclass
class EvalParams:
    data: Data
    device: str
    pretrained: str
    model_type: SupervisedModelsArg
    batch_size: int
    organ: Organ


@dataclass
class EvalResult:
    out: torch.Tensor
    embeddings: torch.Tensor


class EvalRunner:
    def __init__(self, params: EvalParams):
        self.params: EvalParams = params
        self._model: Optional[nn.Module] = None
        self._loader: Optional[DataLoader] = None

        if isinstance(self.model, SIGN_MLP):
            self.params.data = SIGN(self.model.num_layers)(self.params.data)

    @staticmethod
    def new(params: EvalParams) -> "EvalRunner":
        cls = {
            SupervisedModelsArg.sup_graphsage: LoaderRunner,
            SupervisedModelsArg.sup_clustergcn: SamplerRunner,
            SupervisedModelsArg.sup_jumping: SamplerRunner,
            SupervisedModelsArg.sup_clustergcne: SamplerEdgeRunner,
            SupervisedModelsArg.sup_clustergin: SamplerEdgeRunner,
            SupervisedModelsArg.sup_clustergine: SamplerEdgeRunner,
            SupervisedModelsArg.sup_gat: SamplerRunner,
            SupervisedModelsArg.sup_gatv2: SamplerRunner,
            SupervisedModelsArg.sup_graphsaint: SamplerRunner,
            SupervisedModelsArg.sup_shadow: ShaDowRunner,
            SupervisedModelsArg.sup_sign: SIGNRunner,
            SupervisedModelsArg.sup_graphunet: GraphUNetRunner,
            SupervisedModelsArg.sup_graphunet_noskip: GraphUNetRunner,
            SupervisedModelsArg.sup_mlp: MLPRunner,
            SupervisedModelsArg.sup_clustergcn_multilabel: ClusterGCNMultilabelRunner,
            SupervisedModelsArg.sup_clustergcn_multihead: ClusterGCNMultilabelRunner,
        }
        ModelClass = cls[params.model_type]
        return ModelClass(params)

    @property
    def model(self):
        if self._model is None:
            _patch_module_getattr()
            self._model = torch.load(
                self.params.pretrained, map_location=self.params.device
            )
        return self._model

    @property
    def loader(self):
        if self._loader is None:
            self._loader = self.setup_dataloader()
        return self._loader

    @classmethod
    def setup_dataloader(cls):
        raise NotImplementedError(
            f"setup_dataloader not implemented for {cls.__name__}"
        )

    @torch.no_grad()
    def inference(self):
        self.model.eval()
        result = self.model_inference()
        out = result.out
        predicted_labels = out.argmax(dim=-1, keepdim=True).squeeze()
        predicted_labels = predicted_labels.cpu().numpy()
        out = out.cpu().detach().numpy()
        return out, result.embeddings, predicted_labels

    @classmethod
    def model_inference(cls) -> EvalResult:
        """Runs inference across all data in loader."""
        raise NotImplementedError(f"model_inference not implemented for {cls.__name__}")


class LoaderRunner(EvalRunner):
    def setup_dataloader(self):
        eval_loader = NeighborLoader(
            self.params.data,
            num_neighbors=[-1],
            batch_size=self.params.batch_size,
            shuffle=False,
        )
        eval_loader.data.num_nodes = self.params.data.num_nodes
        eval_loader.data.n_id = torch.arange(self.params.data.num_nodes)
        return eval_loader

    def model_inference(self) -> EvalResult:
        out, embeddings = self.model.inference(
            self.params.data.x, self.loader, self.params.device
        )
        return EvalResult(out=out, embeddings=embeddings)


class SamplerRunner(EvalRunner):
    def setup_dataloader(self):
        loader = NeighborLoader(
            self.params.data,
            num_neighbors=[-1],
            batch_size=self.params.batch_size,
            shuffle=False,
        )
        loader.data.num_nodes = self.params.data.num_nodes
        loader.data.n_id = torch.arange(self.params.data.num_nodes)
        return loader

    def model_inference(self) -> EvalResult:
        if isinstance(self.model, GraphSAINT):
            self.model.set_aggr("mean")
        out, embeddings = self.model.inference(
            self.params.data.x, self.loader, self.params.device
        )
        return EvalResult(out=out, embeddings=embeddings)


class SamplerEdgeRunner(SamplerRunner):
    def model_inference(self) -> EvalResult:
        if isinstance(self.model, GraphSAINT):
            self.model.set_aggr("mean")
        out, embeddings = self.model.inference(
            self.params.data.x,
            self.loader,
            self.params.device,
        )
        return EvalResult(out=out, embeddings=embeddings)


class ShaDowRunner(EvalRunner):
    def setup_dataloader(self):
        return ShaDowKHopSampler(
            self.params.data,
            depth=6,
            num_neighbors=12,
            node_idx=None,
            batch_size=self.params.batch_size,
            shuffle=False,
        )

    def model_inference(self) -> EvalResult:
        out = []
        embeddings = []
        for batch in self.loader:
            batch = batch.to(self.params.device)
            batch_out, batch_embed = self.model.inference(
                batch.x, batch.edge_index, batch.batch, batch.root_n_id
            )
            out.append(batch_out)
            embeddings.append(batch_embed)
        out = torch.cat(out, dim=0)
        embeddings = torch.cat(embeddings, dim=0)
        return EvalResult(out=out, embeddings=embeddings)


class SIGNRunner(EvalRunner):
    def setup_dataloader(self):
        return DataLoader(
            range(self.params.data.num_nodes),
            batch_size=self.params.batch_size,
            shuffle=False,
        )

    def model_inference(self):
        out = []
        embeddings = []
        for idx in self.loader:
            eval_x = [self.params.data.x[idx].to(self.params.device)]
            eval_x += [
                self.params.data[f"x{i}"][idx].to(self.params.device)
                for i in range(1, self.model.num_layers + 1)
            ]
            out_i, emb_i = self.model.inference(eval_x)
            out.append(out_i)
            embeddings.append(emb_i)
        out = torch.cat(out, dim=0)
        embeddings = torch.cat(embeddings, dim=0)
        return EvalResult(out=out, embeddings=embeddings)


class GraphUNetRunner(EvalRunner):
    def model_inference(self):
        out = self.model.inference(
            self.params.data.x.to(self.params.device),
            self.params.data.edge_index.to(self.params.device))
        print("In Graph U-Net embeddings are same as outputs")
        return EvalResult(out=out, embeddings=out.detach().cpu().clone())


class MLPRunner(EvalRunner):
    def setup_dataloader(self):
        return DataLoader(
            range(self.params.data.num_nodes),
            batch_size=self.params.batch_size,
            shuffle=False,
        )

    def model_inference(self):
        out = []
        embeddings = []
        for idx in self.loader:
            eval_x = self.params.data.x[idx].to(self.params.device)
            out_i, emb_i = self.model.inference(eval_x)
            out.append(out_i)
            embeddings.append(emb_i)
        out = torch.cat(out, dim=0)
        embeddings = torch.cat(embeddings, dim=0)
        return EvalResult(out=out, embeddings=embeddings)
    
class ClusterGCNMultilabelRunner(SamplerRunner):
    @torch.no_grad()
    def inference(self):
        self.model.eval()
        result = self.model_inference()
        out = result.out
        out_1 = out[:, :-2]
        out_2 = out[:, -2:]
        preds_1 = out_1.argmax(dim=-1)
        preds_2 = out_2.argmax(dim=-1)
        preds_2 = preds_2 * -9 + 9
        preds_1 = preds_1.cpu().numpy()
        preds_2 = preds_2.cpu().numpy()
        out_1 = out_1.cpu().detach().numpy()
        out_2 = out_2.cpu().detach().numpy()

        return out_1, preds_1, out_2, preds_2, result.embeddings
