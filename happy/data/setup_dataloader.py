from torch.utils.data.sampler import BatchSampler, WeightedRandomSampler
from torch.utils.data import DataLoader
from torch_geometric.loader import ClusterData, ClusterLoader, NeighborSampler

from happy.data.samplers import GroupSampler
from happy.data.transforms.collaters import cell_collater, collater


def setup_dataloaders(nuclei, datasets, num_workers, train_batch_size, val_batch_size):
    collate_fn = collater if nuclei else cell_collater

    dataloaders = {}
    for dataset in datasets:
        if dataset == "train":
            dataloaders[dataset] = get_dataloader(
                "train",
                datasets[dataset],
                collate_fn,
                num_workers,
                nuclei,
                train_batch_size,
            )
        else:
            dataloaders[dataset] = get_dataloader(
                "val",
                datasets[dataset],
                collate_fn,
                num_workers,
                nuclei,
                val_batch_size,
            )
    print("Dataloaders configured")
    return dataloaders


def get_dataloader(split, dataset, collater, num_workers, nuclei, batch_size):
    if split == "train" and not nuclei:
        sampler = BatchSampler(
            WeightedRandomSampler(
                dataset.class_sampling_weights, len(dataset), replacement=True
            ),
            batch_size=batch_size,
            drop_last=False,
        )
    else:
        shuffle = True if split == "train" else False
        sampler = GroupSampler(
            dataset, batch_size=batch_size, drop_last=False, shuffle=shuffle
        )
    return DataLoader(
        dataset, num_workers=num_workers, collate_fn=collater, batch_sampler=sampler
    )


def setup_graph_dataloaders(data, batch_size, num_neighbors):
    cluster_data = ClusterData(
        data, num_parts=int(data.x.size()[0] / num_neighbors), recursive=False
    )
    train_loader = ClusterLoader(
        cluster_data, batch_size=batch_size, shuffle=True, num_workers=12
    )
    val_loader = NeighborSampler(
        data.edge_index, sizes=[-1], batch_size=1024, shuffle=False, num_workers=12
    )
    return train_loader, val_loader
