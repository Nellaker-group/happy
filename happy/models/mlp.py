from torch.nn import Linear
import torch.nn.functional as F
import torch
from torch_geometric.nn import norm


class MLP(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout, num_layers):
        super(MLP, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.lins = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else hidden_channels
            self.lins.append(Linear(in_channels, hidden_channels))
            self.bns.append(norm.BatchNorm(hidden_channels))
        self.lins.append(Linear(hidden_channels, out_channels))

    def forward(self, x):
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = F.relu(self.bns[i](x))
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return F.log_softmax(x, dim=-1)

    def inference(self, x):
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = F.relu(self.bns[i](x))
        embeddings = x.detach().clone()
        x = self.lins[-1](x)
        return x, embeddings
