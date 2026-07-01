import torch.nn as nn
from torch.nn import Linear
import torch.nn.functional as F
import torch
from torch_geometric.nn import norm


class SIGN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout, num_layers):
        super(SIGN, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.lins = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        for _ in range(num_layers + 1):
            self.lins.append(Linear(in_channels, hidden_channels))
            self.bns.append(norm.BatchNorm(hidden_channels))
        self.lin = Linear((num_layers + 1) * hidden_channels, out_channels)

    def forward(self, xs):
        hs = []
        for i, (x, lin) in enumerate(zip(xs, self.lins)):
            h = lin(x)
            h = F.relu(self.bns[i](h))
            h = F.dropout(h, p=self.dropout, training=self.training)
            hs.append(h)
        h = torch.cat(hs, dim=-1)
        o = self.lin(h)
        return F.log_softmax(o, dim=-1)

    def inference(self, xs):
        hs = []
        for i, (x, lin) in enumerate(zip(xs, self.lins)):
            h = lin(x)
            h = F.relu(self.bns[i](h))
            hs.append(h)
        h = torch.cat(hs, dim=-1)
        embeddings = h.detach().clone()
        o = self.lin(h)
        return o, embeddings
