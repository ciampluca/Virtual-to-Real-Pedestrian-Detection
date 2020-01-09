import math
import sys
import time
import numpy as np

import torch

import torchvision.models.detection.mask_rcnn

from references.detection.coco_utils import get_coco_api_from_dataset
from .coco_eval import CocoEvaluator
from references.detection import utils


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


@torch.no_grad()
def evaluate(model, data_loader, device, categories=None, save_on_file=None, dataset_name=None, yolo=None,
             max_dets=None):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    # cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types, max_dets=max_dets)

    num_image = 0
    for image, targets in metric_logger.log_every(data_loader, 100, header):
        image = list(img.to(device) for img in image)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        batch_size = len(targets)

        torch.cuda.synchronize()
        model_time = time.time()
        if yolo:
            orig_dims = []
            for _ in range(batch_size):
                if num_image < len(data_loader.dataset):
                    image_width, image_height = data_loader.dataset.imgs_id_dimension[num_image]
                    orig_dims.append((image_height, image_width))
                    num_image += 1
            outputs = model(image, orig_dimensions=orig_dims)
        else:
            outputs = model(image)

        outputs = [{k: v.to(device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time
        # TODO improve
        # Filtering outputs considering only user defined categories
        if categories:
            filtered_outputs = []
            for output in outputs:
                bbs, scores, labels = output['boxes'].data.cpu().numpy(), output['scores'].data.cpu().numpy(), \
                                      output['labels'].data.cpu().numpy()
                filtered_bbs, filtered_scores, filtered_labels = [], [], []
                for counter, label in enumerate(labels):
                    if label in categories:
                        filtered_bbs.append(bbs[counter])
                        filtered_scores.append(scores[counter])
                        filtered_labels.append(labels[counter])
                if not filtered_bbs:
                    filtered_bbs = torch.empty(0, 4).to(device)
                else:
                    filtered_bbs = torch.from_numpy(np.array(filtered_bbs)).to(device)
                outputs = {
                    'boxes': filtered_bbs,
                    'labels': torch.from_numpy(np.array(filtered_labels)).to(device),
                    'scores': torch.from_numpy(np.array(filtered_scores)).to(device),
                }
                filtered_outputs.append(outputs)
            res = {target["image_id"].item(): output for target, output in zip(targets, filtered_outputs)}
        else:
            res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    # coco_evaluator.summarize()
    # TODO improve (better move from here?)
    coco_evaluator.summarize()
    if save_on_file:
        print("Saving results to file")
        if dataset_name:
            print("Validation on dataset: {}".format(dataset_name), file=open(save_on_file, 'a+'))
        for iou_type, coco_eval in coco_evaluator.coco_eval.items():
            print("IoU metric: {}".format(iou_type), file=open(save_on_file, 'a+'))
            print(coco_eval.stats, file=open(save_on_file, 'a+'))
    torch.set_num_threads(n_threads)

    for iou_type, coco_eval in coco_evaluator.coco_eval.items():
        ap_to_return = coco_eval.stats[1]

    return coco_evaluator
