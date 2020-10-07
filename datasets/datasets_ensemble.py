import random
from PIL import ImageDraw

import torch
from torch.utils.data import Dataset, Sampler, DataLoader

from torchvision import transforms as torchvision_transforms

from utils import transforms as custom_T
from datasets.custom_yolo_annotated_dataset import CustomYoloAnnotatedDataset


class DatasetsEnsemble(Dataset):

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
class EnsembleBatchSampler(Sampler):

    def __init__(self, dataset, tgt_imgs_in_batch, batch_size, shuffle=False):
        super().__init__(dataset)

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
        return [
            [source_idxs[x % len(self.dataset.source_dataset)] for x in range(i*num_source_images_per_batch, (i+1)*num_source_images_per_batch)] +
                [target_idxs[x % len(self.dataset.target_dataset)] for x in range(i*num_target_images_per_batch, (i+1)*num_target_images_per_batch)]
                for i in range(0, len(self.dataset.source_dataset) // num_source_images_per_batch)
        ]

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.groups)
        for group in self.groups:
            yield group

    def __len__(self):
        return len(self.groups)


# Testing code
if __name__ == "__main__":
    dataset_root_path_1 = ""
    dataset_root_path_2 = ""
    split = "train"
    percentage = 15
    NUM_WORKERS = 0
    BATCH_SIZE = 4
    TGT_IMAGES_IN_BATCH = 1
    DEVICE = "cuda"
    DATASET_NAME_1 = ""
    DATASET_NAME_2 = ""

    transforms = None
    if split == "train":
        transforms = custom_T.Compose([
            custom_T.RandomHorizontalFlip(),
            custom_T.RandomCrop(),
            custom_T.ToTensor(),
            custom_T.FasterRCNNResizer()
        ])
    elif split == "val" or split == "test":
        transforms = custom_T.Compose([
            custom_T.ToTensor(),
            custom_T.FasterRCNNResizer()
        ])

    dataset_1 = CustomYoloAnnotatedDataset(dataset_root_path_1, transforms=transforms, dataset_name=DATASET_NAME_1,
                                           percentage=percentage, split=split)
    dataset_2 = CustomYoloAnnotatedDataset(dataset_root_path_2, transforms=transforms, dataset_name=DATASET_NAME_2,
                                           percentage=percentage, split=split)

    dataset = DatasetsEnsemble(dataset_1, dataset_2)

    data_loader = DataLoader(
        dataset,
        collate_fn=dataset.source_dataset.standard_collate_fn,
        num_workers=NUM_WORKERS,
        batch_sampler=EnsembleBatchSampler(dataset,
                                           batch_size=BATCH_SIZE,
                                           shuffle=True,
                                           tgt_imgs_in_batch=TGT_IMAGES_IN_BATCH)
    )

    for images, targets in data_loader:
        images = list(image.to(DEVICE) for image in images)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

        for image, target in zip(images, targets):
            img_id = target['image_id'].item()
            print(img_id)
            pil_image = torchvision_transforms.ToPILImage()(image.cpu())
            draw = ImageDraw.Draw(pil_image)
            for bb in target['boxes']:
                draw.rectangle([bb[0].item(), bb[1].item(), bb[2].item(), bb[3].item()])
            pil_image.save("../output_debug/{}.png".format(img_id))
