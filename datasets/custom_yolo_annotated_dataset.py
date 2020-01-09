import os
from PIL import Image
import random
import numpy as np

import torch
import torch.nn.functional as F

from torchvision.datasets import VisionDataset


def resize_image(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image


class CustomYoloAnnotatedDataset(VisionDataset):

    def __init__(self, root, transforms=None, transform=None, target_transform=None, dataset_name=None,
                 multiscale=False, img_size=None, percentage=None, split="train"):
        super().__init__(root, transforms, transform, target_transform)
        # load all image and target files, sorting them to ensure that they are aligned
        assert not (percentage and split == "test"), "Cannot use percentage while testing"

        if split == "val":
            self.images_path = os.path.join(root, "imgs/val")
            self.targets_path = os.path.join(root, "bbs/val")
        elif split == "train":
            self.images_path = os.path.join(root, "imgs/train")
            self.targets_path = os.path.join(root, "bbs/train")
        elif split == "test":
            self.images_path = os.path.join(root, "imgs/test")
            self.targets_path = None
        else:
            raise ValueError("split {} not known".format(split))

        if dataset_name:
            self.dataset_name = dataset_name
        if percentage:
            all_images = sorted([img for img in os.listdir(self.images_path) if img.endswith(".png") or
                                 img.endswith(".jpg") or img.endswith("jpeg")])
            all_targets = sorted([target for target in os.listdir(self.targets_path)
                                  if target.endswith(".txt")])
            num_images = int((len(all_images) / 100) * percentage)
            if split == "train":
                indices = torch.randperm(len(all_images)).tolist()
                indices = indices[-num_images:]
            else:
                indices = range(0, num_images * 200, 200)
                indices = [i % len(all_images) for i in indices]

            self.images = [all_images[i] for i in indices]
            self.targets = [all_targets[i] for i in indices]
        else:
            self.images = list(sorted(os.listdir(self.images_path)))
            self.targets = list(sorted(os.listdir(self.targets_path))) if self.targets_path is not None else None

        self.multiscale = multiscale
        self.batch_count = 0
        # TODO check next operations, some are slow, some useful only in the training phase and only with yolo
        self.imgs_id_name = {k: v for k, v in enumerate(self.images)}
        self.imgs_id_path = [os.path.join(self.images_path, img_name) for k, img_name in self.imgs_id_name.items()]
        # self.imgs_dimensions = []
        # for path in self.images:
        #     image_path = os.path.join(self.images_path, path)
        #     image = Image.open(image_path).convert("RGB")
        #     self.imgs_dimensions.append(image.size)
        # self.imgs_id_dimension = {k: v for k, v in enumerate(self.imgs_dimensions)}
        # if img_size:
        #     self.img_size = img_size
        #     self.min_size = self.img_size - 3 * 32
        #     self.max_size = self.img_size + 3 * 32

    def __getitem__(self, index):
        # load image
        image_path = os.path.join(self.images_path, self.images[index])
        image = Image.open(image_path).convert("RGB")

        image_width, image_height = image.size

        if self.targets is not None:
            # load target; note that bbs are in YOLO format and must be converted in top-left/bottom-right standard
            target_path = os.path.join(self.targets_path, self.targets[index])
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
            labels = torch.ones((num_bbs,), dtype=torch.int64)     # there is only one class
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
        return len(self.images)

    # TODO check, useful for the training phase
    def custom_collate_fn_yolo(self, batch):
        imgs, targets = list(zip(*batch))

        # Add sample index to targets
        for i, boxes in enumerate(targets):
            boxes[:, 0] = i
        targets = torch.cat(targets, 0)

        # Selects new image size every tenth batch
        if self.multiscale and self.batch_count % 10 == 0:
            self.img_size = random.choice(range(self.min_size, self.max_size + 1, 32))

        # Resize images to input shape
        imgs = torch.stack([resize_image(img, self.img_size) for img in imgs])
        self.batch_count += 1

        return imgs, targets

    def standard_collate_fn(self, batch):
        return list(zip(*batch))

