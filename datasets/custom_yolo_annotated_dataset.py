import os
from PIL import Image, ImageDraw
import numpy as np
from collections import OrderedDict
import yaml

import torch
from torch.utils.data import DataLoader

from torchvision.datasets import VisionDataset
from torchvision import transforms as torchvision_transforms

from utils import transforms as custom_T


class CustomYoloAnnotatedDataset(VisionDataset):

    def __init__(self, data_root, dataset_names, transforms=None, percentage=None, phase="train"):
        super().__init__(data_root, transforms)

        assert phase == "train" or phase == "val" or phase == "test", "Not recognized phase"
        assert not (percentage and phase == "test"), "Cannot use percentage while testing"

        self.images = OrderedDict()
        self.targets = OrderedDict()
        self.images_path = OrderedDict()
        self.targets_path = OrderedDict()

        for dataset_name, dataset_cfg in dataset_names.items():
            dataset_root_path, split = dataset_names[dataset_name].rsplit(".", 1)
            assert split == "train" or split == "val" or split == "test", "Not recognized split"
            if (phase == "train" or phase == "val") and split == "test":
                raise AssertionError("Can not use test split in training or val mode")

            if phase == "train" or phase == "val" or (phase == "test" and split != "test"):
                if split == "train":
                    self.images_path[dataset_name] = os.path.join(data_root, dataset_root_path, "imgs/{}".format(split))
                    self.targets_path[dataset_name] = os.path.join(data_root, dataset_root_path, "bbs/{}".format(split))
                elif split == "val":
                    self.images_path[dataset_name] = os.path.join(data_root, dataset_root_path, "imgs/{}".format(split))
                    self.targets_path[dataset_name] = os.path.join(data_root, dataset_root_path, "bbs/{}".format(split))
            else:       # phase == "test" and split == "test"
                self.images_path[dataset_name] = os.path.join(data_root, dataset_root_path, "imgs/{}".format(split))
                self.targets_path[dataset_name] = None

            if percentage:
                all_images = sorted([file for file in os.listdir(self.images_path[dataset_name])
                                     if file.endswith(".jpg") or file.endswith(".png") or file.endswith(".jpeg")])
                all_targets = sorted([file for file in os.listdir(self.targets_path[dataset_name]) if file.endswith(".txt")])
                num_images = int((len(all_images) / 100) * percentage)
                if phase == "train":
                    indices = torch.randperm(len(all_images)).tolist()
                    indices = indices[-num_images:]
                else:
                    assert len(all_images) % 211 != 0, "Validation set must be not multiple of 200"
                    indices = range(0, num_images * 200, 200)
                    indices = [i % len(all_images) for i in indices]
                self.images[dataset_name] = [all_images[i] for i in indices]
                self.targets[dataset_name] = [all_targets[i] for i in indices]
            else:
                self.images[dataset_name] = sorted([file for file in os.listdir(self.images_path[dataset_name])
                                                   if file.endswith(".jpg") or file.endswith(".png") or file.endswith(".jpeg")])
                self.targets[dataset_name] = sorted([file for file in os.listdir(self.targets_path[dataset_name]) if file.endswith(".txt")]) \
                    if self.targets_path[dataset_name] is not None else None

            self.dataset_num_imgs = [len(dataset_img_files) for dataset_img_files in self.images.values()]
            self.dataset_start_index = [sum(self.dataset_num_imgs[:i]) for i in range(len(self.dataset_num_imgs))]
            self.joint_dataset_length = sum(self.dataset_num_imgs)

    def __getitem__(self, index):
        for i, dsi in enumerate(self.dataset_start_index):
            if index >= dsi:
                dataset_name = list(self.targets.keys())[i]
                start_index = dsi

        # load image
        image_path = os.path.join(self.images_path[dataset_name], self.images[dataset_name][index - start_index])
        image = Image.open(image_path).convert("RGB")

        image_width, image_height = image.size

        if self.targets[dataset_name] is not None:
            # load target; from normalized x,y,w,h (x,y of the center) to denormalized xyxy (top left bottom right)
            target_path = os.path.join(self.targets_path[dataset_name], self.targets[dataset_name][index - start_index])
            bounding_boxes, bounding_boxes_areas = [], []
            num_bbs = 0
            with open(target_path, 'r') as bounding_box_file:
                for line in bounding_box_file:
                    x_center = float(line.split()[1]) * image_width
                    y_center = float(line.split()[2]) * image_height
                    bb_width = float(line.split()[3]) * image_width
                    bb_height = float(line.split()[4]) * image_height
                    x_min = x_center - (bb_width / 2.0)
                    x_max = x_min + bb_width
                    y_min = y_center - (bb_height / 2.0)
                    y_max = y_min + bb_height
                    bounding_boxes.append([x_min, y_min, x_max, y_max])
                    area = (y_max - y_min) * (x_max - x_min)
                    bounding_boxes_areas.append(area)
                    num_bbs += 1

            if num_bbs == 0:
                bounding_boxes = [[]]

            # Converting everything related to the target into a torch.Tensor
            bounding_boxes = torch.as_tensor(bounding_boxes, dtype=torch.float32)
            bounding_boxes_areas = torch.as_tensor(bounding_boxes_areas, dtype=torch.float32)
            labels = torch.ones((num_bbs,), dtype=torch.int64)  # there is only one class
            image_id = torch.tensor([index])
            # suppose all instances are not crowd
            iscrowd = torch.zeros((num_bbs,), dtype=torch.int64)

            target = {}
            target["boxes"] = bounding_boxes
            target["labels"] = labels
            target["image_id"] = image_id
            target["area"] = bounding_boxes_areas
            target["iscrowd"] = iscrowd
        else:
            target = None

        image = np.asarray(image)
        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self):
        return self.joint_dataset_length

    def standard_collate_fn(self, batch):
        return list(zip(*batch))


# Testing code
# if __name__ == "__main__":
#     phase = "val"
#     percentage = None
#     NUM_WORKERS = 0
#     BATCH_SIZE = 4
#     DEVICE = "cpu"
#     cfg_file_path = "./cfg/viped_training_resnet50.yaml"
#
#     with open(cfg_file_path, 'r') as stream:
#         try:
#             cfg_file = yaml.safe_load(stream)
#         except yaml.YAMLError as exc:
#             print(exc)
#
#     data_cfg = cfg_file['dataset']
#     data_root = data_cfg['root']
#     if phase == "train":
#         datasets_names = data_cfg['train']
#     elif phase == "val":
#         datasets_names = data_cfg['val']
#     elif phase == "test":
#         datasets_names = data_cfg['test']
#
#     transforms = None
#     if phase == "train":
#         transforms = custom_T.Compose([
#             custom_T.RandomHorizontalFlip(),
#             custom_T.RandomCrop(),
#             custom_T.ToTensor(),
#             custom_T.FasterRCNNResizer()
#         ])
#     elif phase == "val" or phase == "test":
#         transforms = custom_T.Compose([
#             custom_T.ToTensor(),
#             custom_T.FasterRCNNResizer()
#         ])
#
#     # Get dataset and dataloader
#     shuffle = True if phase == "train" else False
#
#     dataset = CustomYoloAnnotatedDataset(data_root, datasets_names, transforms=transforms, percentage=percentage, phase=phase)
#
#     data_loader = DataLoader(
#         dataset,
#         batch_size=BATCH_SIZE,
#         shuffle=shuffle,
#         num_workers=NUM_WORKERS,
#         collate_fn=dataset.standard_collate_fn
#     )
#
#     for images, targets in data_loader:
#         images = list(image.to(DEVICE) for image in images)
#         targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
#
#         for image, target in zip(images, targets):
#             img_id = target['image_id'].item()
#             print(img_id)
#             pil_image = torchvision_transforms.ToPILImage()(image.cpu())
#             draw = ImageDraw.Draw(pil_image)
#             for bb in target['boxes']:
#                 draw.rectangle([bb[0].item(), bb[1].item(), bb[2].item(), bb[3].item()])
#             pil_image.save("./output_debug/{}.png".format(img_id))
