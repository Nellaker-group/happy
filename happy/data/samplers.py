import random

from torch.utils.data.sampler import Sampler


class GroupSampler(Sampler):
    def __init__(self, dataset, batch_size, drop_last, shuffle):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.groups = self.group_images()

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.groups)
        for group in self.groups:
            yield group

    def __len__(self):
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        else:
            return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    def group_images(self):
        # determine the order of the images
        order = list(range(len(self.dataset)))

        if self.shuffle:
            random.shuffle(order)

        # divide into groups, one group = one batch
        groups = []
        for i in range(0, len(order), self.batch_size):
            group = []
            for x in range(i, i+self.batch_size):
                try:
                    group.append(order[x])
                except IndexError:
                    break
            groups.append(group)

        if self.drop_last and len(groups[-1]) != len(groups[0]):
            groups.pop(-1)

        return groups
