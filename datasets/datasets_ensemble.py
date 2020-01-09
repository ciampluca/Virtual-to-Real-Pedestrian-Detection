import os
from PIL import Image
import random
import numpy as np

import torch
from datasets.custom_yolo_annotated_dataset import CustomYoloAnnotatedDataset
import torch.nn.functional as F


class DatasetsEnsemble(torch.utils.data.Dataset):
    def __init__(self, source_dataset, target_dataset):
        self.source_dataset = source_dataset
        self.target_dataset = target_dataset

    def __len__(self):
        return len(self.source_dataset) + len(self.target_dataset)

    def get_source_idxs(self):
        return list(range(len(self.source_dataset)))

    def get_target_idxs(self):
        return list(range(len(self.source_dataset), len(self)))

    def __getitem__(self, index):
        if index < len(self.source_dataset):
            ret = self.source_dataset[index]
            ret[1]["is_source"] = torch.tensor([1])
        else:
            ret = self.target_dataset[index - len(self.source_dataset)]
            ret[1]["is_source"] = torch.tensor([0])
        return ret


# Builds batched with mixed samples from two distinct datasets
# if batch_size = 4 and tgt_imgs_in_batch = 1 -> [s, s, s, t]
class EnsembleBatchSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, tgt_imgs_in_batch, batch_size, shuffle=False):
        assert tgt_imgs_in_batch < batch_size, "Source images in a batch cannot be more than all the images in a batch!"
        self.tgt_imgs_in_batch = tgt_imgs_in_batch
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.dataset = dataset
        self.groups = self.build_groups()

    def build_groups(self):
        num_source_images_per_batch = self.batch_size - self.tgt_imgs_in_batch
        num_target_images_per_batch = self.tgt_imgs_in_batch
        source_idxs = self.dataset.get_source_idxs()
        target_idxs = self.dataset.get_target_idxs()
        if self.shuffle:
            random.shuffle(source_idxs)
            random.shuffle(target_idxs)

        # by default, this sampler constructs batches that, at the end of the epoch, spanned the entire source dataset
        return [[source_idxs[x % len(self.dataset.source_dataset)] for x in range(i*num_source_images_per_batch, (i+1)*num_source_images_per_batch)] +
                [target_idxs[x % len(self.dataset.target_dataset)] for x in range(i*num_target_images_per_batch, (i+1)*num_target_images_per_batch)]
                for i in range(0, len(self.dataset.source_dataset) // num_source_images_per_batch)]

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.groups)
        for group in self.groups:
            yield group

    def __len__(self):
        return len(self.groups)