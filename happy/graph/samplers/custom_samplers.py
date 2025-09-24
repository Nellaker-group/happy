import torch
from torch_cluster import random_walk
from torch_geometric.loader import NeighborSampler


class PosNegNeighborSampler(NeighborSampler):
    def sample(self, batch):
        batch = torch.tensor(batch)
        row, col, _ = self.adj_t.coo()

        # For each node in `batch`, we sample a direct neighbor (as positive
        # example) and a random node (as negative example):
        pos_batch = random_walk(row, col, batch, walk_length=1, coalesced=False)[:, 1]

        neg_batch = torch.randint(
            0, self.adj_t.size(1), (batch.numel(),), dtype=torch.long
        )

        batch = torch.cat([batch, pos_batch, neg_batch], dim=0)
        return super(PosNegNeighborSampler, self).sample(batch)


class CurriculumPosNegNeighborSampler(NeighborSampler):
    def __init__(self, num_negatives, *args, **kwargs):
        self.num_negatives = num_negatives
        super(CurriculumPosNegNeighborSampler, self).__init__(*args, **kwargs)

    def sample(self, batch):
        batch = torch.tensor(batch)
        row, col, _ = self.adj_t.coo()

        # For each node in `batch`, we sample a direct neighbor (as positive
        # example) and a number of random nodes (as negative examples) to choose
        # between for curriculum learning
        pos_batch = random_walk(row, col, batch, walk_length=1, coalesced=False)[:, 1]

        neg_batch = torch.randint(
            0,
            self.adj_t.size(1),
            (batch.numel() * self.num_negatives,),
            dtype=torch.long,
        )

        batch = torch.cat([batch, pos_batch, neg_batch], dim=0)
        return super(CurriculumPosNegNeighborSampler, self).sample(batch)


class SimpleCurriculumPosNegNeighborSampler(NeighborSampler):
    def __init__(self, num_negatives, *args, **kwargs):
        self.num_negatives = num_negatives
        super(SimpleCurriculumPosNegNeighborSampler, self).__init__(*args, **kwargs)

    def sample(self, batch):
        batch = torch.tensor(batch)
        row, col, _ = self.adj_t.coo()

        # For each node in `batch`, we sample a direct neighbor (as positive
        # example) and use other target nodes for calculating the negative examples
        pos_batch = random_walk(row, col, batch, walk_length=1, coalesced=False)[:, 1]

        batch = torch.cat([batch, pos_batch], dim=0)
        return super(SimpleCurriculumPosNegNeighborSampler, self).sample(batch)
