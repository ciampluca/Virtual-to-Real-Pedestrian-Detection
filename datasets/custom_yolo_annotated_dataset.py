import os
from PIL import Image, ImageDraw
import numpy as np

import torch
from torch.utils.data import DataLoader

from torchvision.datasets import VisionDataset
from torchvision import transforms as torchvision_transforms

from utils import transforms as custom_T


class CustomYoloAnnotatedDataset(VisionDataset):

    def __init__(self, root, transforms=None, dataset_name=None, percentage=None, split="train"):
        super().__init__(root, transforms)

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
        return len(self.images)

    def standard_collate_fn(self, batch):
        return list(zip(*batch))


# Testing code
if __name__ == "__main__":
    dataset_root_path = ""
    split = "train"
    percentage = 10
    NUM_WORKERS = 0
    BATCH_SIZE = 2
    DEVICE = "cuda"
    DATASET_NAME = ""

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

    dataset = CustomYoloAnnotatedDataset(dataset_root_path, transforms=transforms, dataset_name=DATASET_NAME,
                                         percentage=percentage, split=split)

    data_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=dataset.standard_collate_fn
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
