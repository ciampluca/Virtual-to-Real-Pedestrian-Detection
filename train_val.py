import math
import sys
import os
from collections import defaultdict
import warnings
from shutil import copyfile
import yaml

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from references.detection import utils
from references.detection.engine import evaluate

from utils import transforms as custom_T
from datasets.custom_yolo_annotated_dataset import CustomYoloAnnotatedDataset
from datasets.datasets_ensemble import EnsembleBatchSampler, DatasetsEnsemble
from models.faster_rcnn import fasterrcnn_resnet50_fpn, fasterrcnn_resnet101_fpn


def get_transform(train=False):
    transforms = []

    if train:
        transforms.append(custom_T.RandomHorizontalFlip())
        transforms.append(custom_T.RandomCrop())

    transforms.append(custom_T.ToTensor())
    transforms.append(custom_T.FasterRCNNResizer())

    return custom_T.Compose(transforms)


def get_model_detection(num_classes, cfg, load_custom_model=False):
    assert cfg['backbone'] == "resnet50" or cfg['backbone'] == "resnet101", "Backbone not supported"

    # replace the classifier with a new one, that has num_classes which is user-defined
    num_classes += 1    # num classes + background

    if load_custom_model:
        model_pretrained = False
        backbone_pretrained = False
    else:
        model_pretrained = cfg['coco_model_pretrained']
        backbone_pretrained = cfg['backbone_pretrained']

    # Creating model
    if cfg['backbone'] == "resnet50":
        model = fasterrcnn_resnet50_fpn(
            pretrained=model_pretrained,
            pretrained_backbone=backbone_pretrained,
            box_detections_per_img=cfg["max_dets_per_image"],
            box_nms_thresh=cfg["nms"],
            box_score_thresh=cfg["det_thresh"],
            model_dir=cfg["cache_folder"],
        )
    else:
        model = fasterrcnn_resnet101_fpn(
            pretrained=model_pretrained,
            pretrained_backbone=backbone_pretrained,
            box_detections_per_img=cfg["max_dets_per_image"],
            box_nms_thresh=cfg["nms"],
            box_score_thresh=cfg["det_thresh"],
            model_dir=cfg["cache_folder"],
        )

    detection_model = model
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    backbone = model.backbone

    return detection_model, backbone


def save_checkpoint(data, path, best_model=None):
    if not os.path.exists(path):
        os.makedirs(path)
    iteration = data['iteration']
    epoch = data['epoch']
    if best_model:
        outfile = 'best_model_{}.pth'.format(best_model)
    else:
        # outfile = 'checkpoint_epoch_{}_iteration_{}.pth'.format(epoch, iteration)
        outfile = 'last_checkpoint.pth'
    outfile = os.path.join(path, outfile)
    torch.save(data, outfile)


def main(args):
    utils.init_distributed_mode(args)
    print(args)

    # Opening YAML cfg config file
    with open(args.cfg_file, 'r') as stream:
        try:
            cfg_file = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    # Retrieving cfg
    train_cfg = cfg_file['training']
    model_cfg = cfg_file['model']
    data_cfg = cfg_file['dataset']

    # Setting device
    device = torch.device(model_cfg['device'])

    # No possible to set checkpoint and pre-trained model at the same time
    if train_cfg['checkpoint'] and train_cfg['pretrained_model']:
        print("You can't set checkpoint and pretrained-model at the same time")
        exit(1)

    # Creating tensorboard writer
    if train_cfg['checkpoint']:
        checkpoint = torch.load(train_cfg['checkpoint'])
        writer = SummaryWriter(log_dir=checkpoint['tensorboard_working_dir'])
    else:
        writer = SummaryWriter(comment="_" + train_cfg['tensorboard_filename'])

    # Saving cfg file in the same folder
    copyfile(args.cfg_file, os.path.join(writer.get_logdir(), os.path.basename(args.cfg_file)))

    #######################
    # Creating model
    #######################
    print("Creating model")
    load_custom_model = False
    if train_cfg['checkpoint'] or train_cfg['pretrained_model']:
        load_custom_model = True
    model, backbone = get_model_detection(num_classes=1, cfg=model_cfg, load_custom_model=load_custom_model)

    # Putting model to device and setting eval mode
    model.to(device)
    model.train()

    # Freeze the backbone parameters, if needed
    if backbone is not None and model_cfg['freeze_backbone']:
        for param in backbone.parameters():
            param.requires_grad = False
        print('Backbone is freezed!')

    #####################################
    # Creating datasets and dataloaders
    #####################################
    data_root = data_cfg['root']

    ################################
    # Creating training datasets and dataloaders
    print("Loading training data")
    train_datasets_names = data_cfg['train']

    if train_cfg['mixed_batch']:
        if train_cfg['tgt_images_in_batch'] <= 0:
            assert train_cfg['tgt_images_in_batch'] > 0, \
                "Using mixed training. You need to specify the tgt_images_in_batch parameter!"
            assert len(train_datasets_names) == 2, "Using mixed training, you need to specify two datasets, " \
                                                   "the first one as the source while the second as the target"
            source_dataset = CustomYoloAnnotatedDataset(
                data_root,
                {train_datasets_names.keys()[0], train_datasets_names.values()[0]},
                transforms=get_transform(train=True),
                phase='train'
            )
            target_dataset = CustomYoloAnnotatedDataset(
                data_root,
                {train_datasets_names.keys()[1], train_datasets_names.values()[1]},
                transforms=get_transform(train=True),
                phase='train'
            )
            train_dataset = DatasetsEnsemble(source_dataset=source_dataset, target_dataset=target_dataset)
            train_dataloader = DataLoader(
                train_dataset,
                collate_fn=train_dataset.source_dataset.standard_collate_fn,
                num_workers=train_cfg['num_workers'],
                batch_sampler=EnsembleBatchSampler(train_dataset,
                                                   batch_size=train_cfg['batch_size'],
                                                   shuffle=True,
                                                   tgt_imgs_in_batch=train_cfg['tgt_images_in_batch'])
            )
            print('Using mixed training datasets. Source: {}, Target: {}. In every batch, {}/{} are from {}'.format(
                train_datasets_names.keys()[0], train_datasets_names.keys()[1], train_cfg['tgt_images_in_batch'],
                train_cfg['batch_size'], train_datasets_names.keys()[1]
            ))
    else:
        train_dataset = CustomYoloAnnotatedDataset(
            data_root,
            train_datasets_names,
            transforms=get_transform(train=True),
            phase='train'
        )
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=train_cfg['batch_size'],
            shuffle=False,
            num_workers=train_cfg['num_workers'],
            collate_fn=train_dataset.standard_collate_fn
        )

    ###############################
    # Creating validation datasets
    print("Loading validation data")
    val_datasets_names = data_cfg['val']

    # Creating dataset(s) and dataloader(s)
    val_dataloaders = dict()
    best_validation_ap = defaultdict(float)
    for dataset_name, dataset_cfg in val_datasets_names.items():
        val_dataset = CustomYoloAnnotatedDataset(
            data_root,
            {dataset_name: dataset_cfg},
            transforms=get_transform(),
            phase="val",
            percentage=train_cfg["percentage_val"]
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=train_cfg['batch_size'],
            shuffle=False,
            num_workers=train_cfg['num_workers'],
            collate_fn=val_dataset.standard_collate_fn
        )
        # Adding created dataloader
        val_dataloaders[dataset_name] = val_dataloader
        # Initializing best validation ap value
        best_validation_ap[dataset_name] = 0.0


    #######################################
    # Defining optimizer and LR scheduler
    #######################################
    ##########################
    # Constructing an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params,
                                lr=train_cfg['lr'],
                                momentum=train_cfg['momentum'],
                                weight_decay=train_cfg['weight_decay'],
                                )

    # and a learning rate scheduler
    if model_cfg['coco_model_pretrained']:
        lr_step_size = min(30000, len(train_dataset))
    else:
        lr_step_size = min(50000, 2*len(train_dataset))
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=lr_step_size,
                                                   gamma=train_cfg['lr_gamma']
                                                   )

    # Defining a warm-up lr scheduler
    warmup_iters = min(1000, len(train_dataloader) - 1)
    warmup_factor = 1. / 1000
    warmup_lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    #############################
    # Resuming a model
    #############################
    start_epoch = 0
    train_step = -1
    # Eventually resuming a pre-trained model
    if train_cfg['pretrained_model']:
        print("Resuming pre-trained model")
        pre_trained_model = torch.load(train_cfg['pretrained_model'])
        model.load_state_dict(pre_trained_model['model'])

    # Eventually resuming from a saved checkpoint
    if train_cfg['checkpoint']:
        print("Resuming from a checkpoint")
        checkpoint = torch.load(train_cfg['checkpoint'])
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        warmup_lr_scheduler.load_state_dict(checkpoint['warmup_lr_scheduler'])
        start_epoch = checkpoint['epoch']
        train_step = checkpoint['iteration']
        for elem_name, elem in checkpoint.items():
            if elem_name.startswith("best_"):
                d_name = elem_name.split("_")[1]
                if d_name in best_validation_ap:
                    best_validation_ap[d_name] = elem
                else:
                    warnings.warn("The dataset {} was not used in the previous training".format(d_name))
                    best_validation_ap[d_name] = 0.0

    ################
    ################
    # Training
    print("Start training")
    for epoch in range(start_epoch, train_cfg['epochs']):
        model.train()
        metric_logger = utils.MetricLogger(delimiter="  ")
        metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        header = 'Epoch: [{}]'.format(epoch)

        for images, targets in metric_logger.log_every(train_dataloader, print_freq=train_cfg['print_freq'], header=header):
            train_step += 1
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)

            losses = sum(loss for loss in loss_dict.values())

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = utils.reduce_dict(loss_dict)
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())

            loss_value = losses_reduced.item()

            if not math.isfinite(loss_value):
                for target in targets:
                    image_id = target['image_id'].item()
                    print(train_dataset.images[image_id])
                print("Loss is {}, stopping training".format(loss_value))
                print(loss_dict_reduced)
                sys.exit(1)

            optimizer.zero_grad()
            losses.backward()
            # clip norm
            torch.nn.utils.clip_grad_norm_(model.parameters(), 50)
            optimizer.step()

            if epoch == 0 and train_step < warmup_iters:
                warmup_lr_scheduler.step()

            metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])

            if train_step % train_cfg['log_loss'] == 0:
                writer.add_scalar('Training/Learning Rate', optimizer.param_groups[0]["lr"], train_step)
                writer.add_scalar('Training/Reduced Sum Losses', losses_reduced, train_step)
                writer.add_scalars('Training/All Losses', loss_dict, train_step)

            if (train_step % train_cfg['save_freq'] == 0 and train_step != 0) \
                    or ((train_cfg['pretrained_model'] or model_cfg['coco_model_pretrained']) and
                        train_step < 6 * train_cfg['save_freq'] and train_step % 200 == 0 and train_step != 0):
                # Validation
                for val_name, val_dataloader in val_dataloaders.items():
                    print("Validation on {}".format(val_name))
                    coco_evaluator = evaluate(model, val_dataloader, device=device, max_dets=model_cfg["max_dets_per_image"])
                    ap = None
                    for iou_type, coco_eval in coco_evaluator.coco_eval.items():
                        ap = coco_eval.stats[1]
                    writer.add_scalar('COCO mAP Validation/{}'.format(val_name), ap, train_step)

                    # Eventually saving best model
                    if ap > best_validation_ap[val_name]:
                        best_validation_ap[val_name] = ap
                        save_checkpoint({
                            'model': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'lr_scheduler': lr_scheduler.state_dict(),
                            'warmup_lr_scheduler':
                                warmup_lr_scheduler.state_dict() if warmup_lr_scheduler is not None else None,
                            'epoch': epoch,
                            'iteration': train_step,
                            'best_{}_ap'.format(val_name): best_validation_ap[val_name],
                        }, writer.get_logdir(), best_model=val_name)

                # Saving last model
                checkpoint_dict = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'warmup_lr_scheduler':
                        warmup_lr_scheduler.state_dict() if warmup_lr_scheduler is not None else None,
                    'epoch': epoch,
                    'iteration': train_step,
                    'tensorboard_working_dir': writer.get_logdir(),
                }
                for d_name, _ in val_dataloaders.items():
                    checkpoint_dict["best_{}_ap".format(d_name)] = best_validation_ap[d_name]
                save_checkpoint(
                    checkpoint_dict,
                    writer.get_logdir())

                # Setting again to train mode
                model.train()

            # Updating lr scheduler
            lr_scheduler.step()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--cfg-file', required=True, help="YAML config file path")

    args = parser.parse_args()

    main(args)

