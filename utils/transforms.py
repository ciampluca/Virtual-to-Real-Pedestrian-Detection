from imgaug import augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox

import torch
from torchvision.models.detection.transform import resize_boxes
from torchvision.transforms import functional as F


class Compose(object):

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)

        return image, target


class BaseImgAugTransform:

    def __call__(self, np_image, target=None):
        if target is not None and target['boxes'].nelement() != 0:
            bboxes = [BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2) for x1, y1, x2, y2 in target['boxes'][:, :4]]
            image_aug, boxes_aug = self.seq(image=np_image, bounding_boxes=bboxes)
            boxes_aug = [[b.x1, b.y1, b.x2, b.y2] for b in boxes_aug]
            target['boxes'][:, :4] = torch.tensor(boxes_aug)
        else:
            image_aug = self.seq(image=np_image)

        return image_aug, target


class RandomHorizontalFlip(BaseImgAugTransform):

    def __init__(self):
        self.seq = iaa.Sequential([
            iaa.Fliplr(0.5)
        ])


class RandomCrop(BaseImgAugTransform):
    def __init__(self):
        self.seq = iaa.Sequential([
            iaa.Crop(percent=(0, 0.2))
        ])


class ToTensor(object):

    def __call__(self, image, target=None):
        image = F.to_tensor(image)

        # Handle images with less than three channels
        if len(image.shape) != 3:
            print("Image not having 3 channels")
            image = image.unsqueeze(0)
            image = image.expand((3, image.shape[1:]))

        return image, target


class FasterRCNNResizer(object):
    """Convert ndarrays in sample to Tensors.
    NOTE: torchvision 0.3 already comes with a resizer, but it is embedded into the model. This obliges the model to fully
    load every image into GPU. Instead, this transformer pre-processes the image resizing it before loading onto
    the GPU.
    """

    def __init__(self, min_side=800, max_side=1333):
        self.min_side = min_side
        self.max_side = max_side

    def __call__(self, image, target):
        h, w = image.shape[-2:]
        min_size = float(min(image.shape[-2:]))
        max_size = float(max(image.shape[-2:]))

        size = self.min_side
        scale_factor = size / min_size
        if max_size * scale_factor > self.max_side:
            scale_factor = self.max_side / max_size
        image = torch.nn.functional.interpolate(
            image[None], scale_factor=scale_factor, mode='bilinear', align_corners=False)[0]

        if target is None or target["boxes"].nelement() == 0:
            return image, target

        bbox = target["boxes"]
        bbox = resize_boxes(bbox, (h, w), image.shape[-2:])
        target["boxes"] = bbox

        return image, target
