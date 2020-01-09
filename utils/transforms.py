import numpy as np
from imgaug import augmenters as iaa
from imgaug import pad_to_aspect_ratio
from imgaug.augmentables.bbs import BoundingBox
from torchvision.models.detection.transform import resize_boxes

import torch

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
        # np_image = image.permute(1, 2, 0).numpy()
        if target is not None and target['boxes'].nelement() != 0:
            bboxes = [BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2) for x1, y1, x2, y2 in target['boxes'][:, :4]]
            image_aug, boxes_aug = self.seq(image=np_image, bounding_boxes=bboxes)
            # image_aug = torch.from_numpy(np.ascontiguousarray(image_aug)).permute(2, 0, 1)
            boxes_aug = [[b.x1, b.y1, b.x2, b.y2] for b in boxes_aug]
            target['boxes'][:, :4] = torch.tensor(boxes_aug)
        else:
            image_aug = self.seq(image=np_image)
            # image_aug = torch.from_numpy(np.ascontiguousarray(image_aug)).permute(2, 0, 1)

        return image_aug, target


# class DirtyCameraLens(BaseImgAugTransform):
#
#     def __init__(self):
#         self.seq = iaa.Sequential([
#             # Small gaussian blur and a random amount of motion blur
#             iaa.GaussianBlur(sigma=(0, 0.5)),
#             iaa.MotionBlur(k=[10, 20], angle=[-45, 45]),
#             # Add some bloom effect
#             iaa.Alpha([0.25, 0.35, 0.55], iaa.Sequential([
#                 iaa.GaussianBlur(sigma=(60, 100)),
#                 iaa.LinearContrast((1, 3)),
#                 iaa.Add((0, 50))
#             ])),
#             # Final contrast adjustment
#             iaa.LinearContrast(alpha=(0.5, 1.5)),
#             # Some lens cloud-like dirtiness
#             iaa.Alpha([0.25, 0.35], iaa.Clouds()),
#         ], random_order=False)


class DirtyCameraLens(BaseImgAugTransform):

    def __init__(self):
        self.seq = iaa.Sequential([
            # iaa.Fliplr(0.5), # horizontal flips
            # Small gaussian blur with random sigma between 0 and 0.5.
            # But we only blur about 50% of all images.
            iaa.GaussianBlur(sigma=(0, 0.5)),
            iaa.MotionBlur(k=[5, 12], angle=[-45, 45]),
            # Strengthen or weaken the contrast in each image.
            iaa.Alpha([0.25, 0.35, 0.55], iaa.Sequential([
                iaa.GaussianBlur(sigma=(60, 100)),
                iaa.LinearContrast((1, 3)),
                iaa.Add((0, 30))
            ])),
            #iaa.Lambda(radial_blur),
            # Add gaussian noise.
            # For 50% of all images, we sample the noise once per pixel.
            # For the other 50% of all images, we sample the noise per pixel AND
            # channel. This can change the color (not only brightness) of the
            # pixels.
            iaa.LinearContrast((0.5, 1.0)),
            iaa.MultiplyHueAndSaturation((0.5, 1.5))
            # iaa.Alpha([0.25, 0.35], iaa.Clouds()),
        ], random_order=False)


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


class PadToSquare(BaseImgAugTransform):

    def __call__(self, np_image, target=None):
        # np_image = image.permute(1, 2, 0).numpy()
        if target is not None and target['boxes'].nelement() != 0:
            bboxes = [BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2) for x1, y1, x2, y2 in target['boxes'][:, :4]]
            image_aug, pad_amounts = pad_to_aspect_ratio(np_image, 1.0, return_pad_amounts=True)
            boxes_aug = [[b.x1 + pad_amounts[3], b.y1 + pad_amounts[0], b.x2 + pad_amounts[3], b.y2 + pad_amounts[0]]
                         for b in bboxes]
            # image_aug = torch.from_numpy(np.ascontiguousarray(image_aug)).permute(2, 0, 1)
            target['boxes'][:, :4] = torch.tensor(boxes_aug)
        else:
            image_aug, pad_amounts = pad_to_aspect_ratio(np_image, 1.0, return_pad_amounts=True)
            # image_aug = torch.from_numpy(np.ascontiguousarray(image_aug)).permute(2, 0, 1)

        return image_aug, target


class Resize(BaseImgAugTransform):

    def __init__(self, img_size):
        self.seq = iaa.Sequential([
            iaa.Resize(size=img_size, interpolation="nearest")
        ])


# TODO check, useful for the training phase
class ToYoloFormat(object):

    def __call__(self, image, target):
        _, h_image, w_image = image.shape
        bbs, labels = target['boxes'].data.cpu().numpy(), target['labels'].data.cpu().numpy()
        labels -= 1     # indexes of yolo classes are shifted by one
        if labels.size != 0:
            labels = labels[np.newaxis].T
            bbs_yolo_format = np.concatenate((labels, bbs), axis=1).astype("float32")
            # Returns (x_center, y_center, w, h)
            x_centers = (bbs_yolo_format[:, 1] + bbs_yolo_format[:, 3]) / 2
            y_centers = (bbs_yolo_format[:, 2] + bbs_yolo_format[:, 4]) / 2
            bb_widths = bbs_yolo_format[:, 3] - bbs_yolo_format[:, 1]
            bb_heights = bbs_yolo_format[:, 4] - bbs_yolo_format[:, 2]
            bbs_yolo_format[:, 1] = x_centers / w_image
            bbs_yolo_format[:, 2] = y_centers / h_image
            bbs_yolo_format[:, 3] = bb_widths / w_image
            bbs_yolo_format[:, 4] = bb_heights / h_image
            bbs_yolo_format = torch.from_numpy(np.ascontiguousarray(bbs_yolo_format))

            targets = torch.zeros((len(bbs_yolo_format), 6))
            targets[:, 1:] = bbs_yolo_format
        else:
            targets = torch.empty((0, 6))

        return image, targets


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

        if target is None:
            return image, target

        bbox = target["boxes"]
        bbox = resize_boxes(bbox, (h, w), image.shape[-2:])
        target["boxes"] = bbox

        return image, target

