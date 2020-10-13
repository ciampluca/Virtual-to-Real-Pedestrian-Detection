import math
import sys
import os
from collections import defaultdict

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from references.detection import utils
from references.detection.engine import evaluate

from utils import transforms as custom_T
from datasets.custom_yolo_annotated_dataset import CustomYoloAnnotatedDataset
from datasets.datasets_ensemble import EnsembleBatchSampler, DatasetsEnsemble


DATASETS = {
    "viped": "./data/viped",
    "MOT17Det": "./data/MOT17Det",
    "MOT20Det": "./data/MOT20Det",
    "COCOPersons": "./data/COCOPersons",
}


def get_dataset(name, transforms, percentage=None, split="train"):
    if name in DATASETS:
        dataset_root_path = DATASETS[name]
        dataset = CustomYoloAnnotatedDataset(dataset_root_path, transforms, dataset_name=name, percentage=percentage,
                                             split=split)
        # Creating a file taking track of the percentage for the val split, and eventually remove the cached dataset
        cache_path = 'dataset_cache'
        txt_file = os.path.join(cache_path, name + ".txt")
        if split == "val":
            if not percentage:
                percentage = 100
            if os.path.isfile(txt_file):
                with open(txt_file) as f:
                    content = f.readlines()
                content = [x.strip() for x in content]
                saved_percentage = int(content[0])
                if percentage != saved_percentage:
                    with open(txt_file, "w") as text_file:
                        text_file.write("{}".format(percentage))
                    if os.path.isfile(os.path.join(cache_path, name + ".pkl")):
                        os.remove(os.path.join(cache_path, name + ".pkl"))
            else:
                if os.path.isfile(os.path.join(cache_path, name + ".pkl")):
                    os.remove(os.path.join(cache_path, name + ".pkl"))
                with open(txt_file, "w") as text_file:
                    text_file.write("{}".format(percentage))
    else:
        raise ValueError("Non existing dataset")

    return dataset


def get_transform(train=False):
    transforms = []

    if train:
        transforms.append(custom_T.RandomHorizontalFlip())
        transforms.append(custom_T.RandomCrop())

    transforms.append(custom_T.ToTensor())
    transforms.append(custom_T.FasterRCNNResizer())

    return custom_T.Compose(transforms)


def get_model_detection(num_classes, args):
    # replace the classifier with a new one, that has num_classes which is user-defined
    num_classes = num_classes + 1  # 1 class (person) + background
    backbone = None

    print('Initializing FasterRCNN detector...')
    # load a model pre-trained eventually pre-trained on COCO; default thresh: 0.05
    faster_rcnn_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=args.pretrained,
                                                                             box_detections_per_img=args.max_dets,
                                                                             box_score_thresh=args.thresh)
    detection_model = faster_rcnn_model
    # get number of input features for the classifier
    in_features = faster_rcnn_model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    faster_rcnn_model.roi_heads.box_predictor = FastRCNNPredictor(in_features,
                                                                  num_classes)
    backbone = faster_rcnn_model.backbone

    return detection_model, backbone


def save_checkpoint(data, path, best_model=None):
    if not os.path.exists(path):
        os.makedirs(path)
    iteration = data['iteration']
    epoch = data['epoch']
    if best_model:
        outfile = 'best_model_{}.pth'.format(best_model)
    else:
        outfile = 'checkpoint_epoch_{}_iteration_{}.pth'.format(epoch, iteration)
    outfile = os.path.join(path, outfile)
    torch.save(data, outfile)


def main(args):
    utils.init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)

    # Creating tensorboard writer
    writer = SummaryWriter(comment="_" + args.tensorboard_file_name)

    ####################
    # Creating model
    print("Creating model")
    model, backbone = get_model_detection(num_classes=1, args=args)

    # Putting model to device and setting eval mode
    model.to(device)
    model.train()

    # Eventually resuming a pre-trained model
    if args.resume:
        print("Resuming pre-trained model")
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model'])

    # Freeze the backbone parameters, if needed
    if backbone is not None and args.freeze_backbone:
        for param in backbone.parameters():
            param.requires_grad = False
        print('Backbone is freezed!')

    ##########################
    # Construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params,
                                lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                )

    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=args.lr_step_size,
                                                   gamma=args.lr_gamma
                                                   )

    ################################
    # Creating training datasets
    print("Loading training data")
    train_datasets_dict = {
        'viped': lambda: get_dataset("viped", get_transform(train=True)),
        'MOT20Det': lambda: get_dataset("MOT20Det", get_transform(train=True)),
        'MOT17Det': lambda: get_dataset("MOT17Det", get_transform(train=True)),
        'COCOPersons': lambda: get_dataset("COCOPersons", get_transform(train=True)),
    }

    # Preparing training dataloader
    if args.train_on in train_datasets_dict:
        # the train dataset is a normal single dataset
        train_dataset = train_datasets_dict[args.train_on]()
        train_dataloader = DataLoader(
                            train_dataset,
                            batch_size=args.batch_size,
                            shuffle=True,
                            num_workers=args.workers,
                            collate_fn=train_dataset.standard_collate_fn
                          )
        print('Using training dataset: {}'.format(args.train_on))
    elif ',' in args.train_on:
        assert args.tgt_images_in_batch > 0, "Using mixed training. " \
                                             "You need to specify the args.tgt_images_in_batch parameter!"
        # the train dataset is an ensemble of datasets
        source_dataset_name, target_dataset_name = args.train_on.split(',')
        train_dataset = DatasetsEnsemble(train_datasets_dict[source_dataset_name](),
                                         train_datasets_dict[target_dataset_name]())
        train_dataloader = DataLoader(
            train_dataset,
            collate_fn=train_dataset.source_dataset.standard_collate_fn,
            num_workers=args.workers,
            batch_sampler=EnsembleBatchSampler(train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               tgt_imgs_in_batch=args.tgt_images_in_batch)
        )
        print('Using mixed training datasets. Source: {}, Target: {}. In every batch, {}/{} are from {}'.format(
            source_dataset_name, target_dataset_name, args.tgt_images_in_batch, args.batch_size, target_dataset_name
        ))
    else:
        raise ValueError('Dataset not known!')

    ###############################
    # Creating validation datasets
    print("Loading validation data")
    val_datasets_dict = {
        'viped': lambda: get_dataset("viped", get_transform(train=False), split="val"),
        'MOT20Det': lambda: get_dataset("MOT20Det", get_transform(train=False), split="val"),
        'MOT17Det': lambda: get_dataset("MOT17Det", get_transform(train=False), split="val"),
    }

    # Creating val dataloaders
    val_dataloaders = dict()
    if args.validate_on == "all":
        for d_name, dataset in val_datasets_dict.items():
            val_dataset = val_datasets_dict[d_name]()
            val_data_loader = DataLoader(
                val_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.workers,
                collate_fn=val_dataset.standard_collate_fn
            )
            val_dataloaders[d_name] = val_data_loader
    elif args.validate_on in val_datasets_dict:
        val_dataset = val_datasets_dict[args.validate_on]()
        val_data_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            collate_fn=val_dataset.standard_collate_fn
        )
        val_dataloaders[args.validate_on] = val_data_loader
    else:
        raise ValueError('Validation dataset not known!')

    # Initializing best validation ap values
    best_validation_ap = defaultdict(float)
    for d_name, dataset in val_datasets_dict.items():
        best_validation_ap[d_name] = 0.0

    ####################################
    # Defining a warm-up lr scheduler
    warmup_iters = min(1000, len(train_dataloader) - 1)
    warmup_factor = 1. / 1000
    warmup_lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    ################
    ################
    # Training
    print("Start training")
    start_epoch = 0
    train_step = -1
    for epoch in range(start_epoch, args.epochs):
        model.train()
        metric_logger = utils.MetricLogger(delimiter="  ")
        metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        header = 'Epoch: [{}]'.format(epoch)

        for images, targets in metric_logger.log_every(train_dataloader, print_freq=args.print_freq, header=header):
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
                print("Loss is {}, stopping training".format(loss_value))
                print(loss_dict_reduced)
                sys.exit(1)

            optimizer.zero_grad()
            losses.backward()
            # clip norm
            torch.nn.utils.clip_grad_norm(model.parameters(), 50)
            optimizer.step()

            if epoch == 0 and train_step < warmup_iters:
                warmup_lr_scheduler.step()

            metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])

            if train_step % args.log_loss == 0:
                writer.add_scalar('Training/Learning Rate', optimizer.param_groups[0]["lr"], train_step)
                writer.add_scalar('Training/Reduced Sum Losses', losses_reduced, train_step)
                writer.add_scalars('Training/All Losses', loss_dict, train_step)

            if (train_step % args.save_freq == 0 and train_step != 0) \
                    or (args.pretrained and train_step < 5 * args.save_freq and train_step % 200 == 0 and train_step != 0) \
                    or train_step == 100:
                # Validation
                for val_name, val_dataloader in val_dataloaders.items():
                    print("Validation on {}".format(val_name))
                    coco_evaluator = evaluate(model, val_dataloader, device=device, max_dets=args.max_dets)
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
                save_checkpoint({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'warmup_lr_scheduler':
                        warmup_lr_scheduler.state_dict() if warmup_lr_scheduler is not None else None,
                    'epoch': epoch,
                    'iteration': train_step,
                }, writer.get_logdir())

                # Setting again to train mode
                model.train()

            # Updating lr scheduler
            lr_scheduler.step()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda', help='Device. Default is cuda')
    parser.add_argument('--thresh', default=0.05, type=float, help="Box score threshold (default 0.05)")
    parser.add_argument('--max-dets', default=350, type=int, help="Max num of detections per image")
    parser.add_argument('--pretrained', default=True, help="coco pretrained")
    parser.add_argument('--freeze-backbone', default=False, help="Freeze the backbone during train")
    parser.add_argument('--lr', default=0.005, type=float,
                        help='Initial learning rate, 0.005 is the default value for training (fine tuning)')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=5e-4, type=float, metavar='W',
                        help='weight decay (default: 1e-4)', dest='weight_decay')
    parser.add_argument('--lr-step-size', default=20000, type=int, help='Decrease lr every step-size epochs')
    parser.add_argument('--lr-gamma', default=0.1, type=float, help='Decrease lr by a factor of lr-gamma')
    parser.add_argument('--train-on', default='viped', type=str,
                        help="Which dataset use for training. Possible values are viped (default), MOT17Det, "
                             "MOT20Det, COCOPersons. " 
                             "You can also put two single names separated by a comma (e.g., viped,MOT17Det), and in "
                             "this case the Mixed Batch DA approach is automatically selected")
    parser.add_argument('-b', '--batch-size', default=4, type=int, help='Batch size')
    parser.add_argument('--epochs', default=50, type=int, metavar='N',
                        help='Number of total epochs to run (default: 50)')
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='Number of data loading workers (default: 8)')
    parser.add_argument('--tgt-images-in-batch', default=1, type=int,
                        help="In case of mixed batches, how many target images in the batch (default: 1)")
    parser.add_argument('--validate-on', default='all', type=str,
                        help="Which dataset use for validation. Possible values are all (default), viped, MOT17Det, "
                             "MOT20Det.")
    parser.add_argument('--save-freq', default=1000, type=int, help='Save frequency')
    parser.add_argument('--log-loss', default=10, type=int, help='Save loss values using tensorboard (frequency)')
    parser.add_argument('--print-freq', default=100, type=int, help='print frequency')
    parser.add_argument('--tensorboard-file-name', default="default_experiment_name",
                        help='name of the tensorboard file')
    parser.add_argument('--resume', default='', help='load a pre-trained model')

    args = parser.parse_args()

    print('Using pretrained model: {}'.format(args.pretrained))
    if args.pretrained:
        args.lr_step_size /= 4
    print('Learning rate step size: {}'.format(args.lr_step_size))

    main(args)

