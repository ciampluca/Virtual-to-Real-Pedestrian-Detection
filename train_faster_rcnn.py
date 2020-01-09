import sys
import math
import os

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from references.detection import utils
from references.detection.engine import evaluate

from utils import transforms as custom_T
from datasets.custom_yolo_annotated_dataset import CustomYoloAnnotatedDataset
from datasets.datasets_ensemble import EnsembleBatchSampler, DatasetsEnsemble

from models.fasterrcnn_mmd import MMDFasterRCNN

DATASETS = {
    "viped": "/media/luca/Dati_1_SSD/datasets/pedestrian_detection/ViPeD",
    "mot19": "/media/luca/Dati_1_SSD/datasets/pedestrian_detection/MOT19Det",
    "mot17": "/media/luca/Dati_1_SSD/datasets/pedestrian_detection/MOT17Det",
    "crowd_human": "/media/luca/Dati_1_SSD/datasets/pedestrian_detection/crowd_human",
    "city_persons": "/media/luca/Dati_1_SSD/datasets/pedestrian_detection/city_persons",
    "COCO_persons": "/media/luca/datino/pedestrian_detection/COCO_Persons",
}

TENSORBOARD_RESULT_FILE_NAME = "faster_rcnn_viped_basic_cocoFineTuning_maxDets350_thresh05"


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


def get_dataset(name, transform, percentage=None, img_size=None, val=False):
    if name in DATASETS:
        dataset_root_path = DATASETS[name]
        dataset = CustomYoloAnnotatedDataset(dataset_root_path, transform, dataset_name=name, img_size=img_size,
                                             percentage=percentage, val=val)
    else:
        raise ValueError("Non existing dataset")

    return dataset


def get_transform(train=False, yolo=False, aug=None):
    assert aug == 'dirty_camera_lens' or aug == 'gan' or aug is None, "Aug parameter not valid"

    transforms = []

    if yolo:
        transforms.append(custom_T.PadToSquare())
        transforms.append(custom_T.Resize(img_size=None))

    if train:
        transforms.append(custom_T.RandomHorizontalFlip())
        transforms.append(custom_T.RandomCrop())

    if aug == 'dirty_camera_lens':
        print("Augmentation: Dirty Camera Lens")
        transforms.append(custom_T.DirtyCameraLens())

    transforms.append(custom_T.ToTensor())
    transforms.append(custom_T.FasterRCNNResizer())

    return custom_T.Compose(transforms)


def get_model_detection(num_classes, model, pretrained=True):
    # replace the classifier with a new one, that has num_classes which is user-defined
    num_classes = 2  # 1 class (person) + background
    backbone = None

    if 'fasterrcnn' in model:
        print('Initializing FasterRCNN detector...')
        # load a model pre-trained pre-trained on COCO; default thresh: 0.05
        faster_rcnn_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=pretrained, box_detections_per_img=args.max_dets,
                                                                     box_score_thresh=args.thresh)
        detection_model = faster_rcnn_model
        # get number of input features for the classifier
        in_features = faster_rcnn_model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        faster_rcnn_model.roi_heads.box_predictor = FastRCNNPredictor(in_features,
                                                                      num_classes)
        if model == 'fasterrcnn_mmd':
            assert args.tgt_images_in_batch > 0, 'Argument tgt_images_in_batch should be > 0'
            print('Using MMD domain adaptation')
            detection_model = MMDFasterRCNN(faster_rcnn_model,
                                              src_imgs_in_batch=args.batch_size - args.tgt_images_in_batch,
                                              thresh=args.thresh, pretrained=pretrained)
        backbone = faster_rcnn_model.backbone
    else:
        raise ValueError('Model {} does not exist'.format(model))

    return detection_model, backbone


def main(args):
    utils.init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)

    # Creating tensorboard writer
    if not args.resume:
        writer = SummaryWriter(comment=TENSORBOARD_RESULT_FILE_NAME)
    else:
        writer = SummaryWriter(
            "")

    ######################
    # Creating test data #
    ######################
    print("Loading test data")

    viped_dataset_test = get_dataset("viped", get_transform(train=False, aug=args.aug), percentage=5, val=True)
    mot19_dataset_test = get_dataset("mot19", get_transform(train=False), val=True)
    mot17_dataset_test = get_dataset("mot17", get_transform(train=False), val=True)
    crowd_human_dataset_test = get_dataset("crowd_human", get_transform(train=False), val=True)
    city_persons_dataset_test = get_dataset("city_persons", get_transform(train=False), val=True)
    coco_persons_dataset_test = get_dataset("COCO_persons", get_transform(train=False), val=True)

    ##########################
    # Creating training data #
    ##########################
    print("Loading training data")
    train_datasets_dict = {
        'viped': lambda: get_dataset("viped", get_transform(train=True, aug=args.aug)),
        'mot19': lambda: get_dataset("mot19", get_transform(train=True)),
        'mot17': lambda: get_dataset("mot17", get_transform(train=True)),
        'crowd_human': lambda: get_dataset("crowd_human", get_transform(train=True)),
        'city_persons': lambda: get_dataset("city_persons", get_transform(train=True)),
        'COCO_persons:': lambda: get_dataset("COCO_persons", get_transform(train=True)),
    }

    #################################
    # Preparing training dataloader #
    #################################
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
        # the train dataset is an ensamble of datasets
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

    ##############################
    # Preparing test dataloaders #
    ##############################

    data_loader_viped_test = DataLoader(
        viped_dataset_test,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=viped_dataset_test.standard_collate_fn
    )

    data_loader_mot19_test = DataLoader(
        mot19_dataset_test,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=mot19_dataset_test.standard_collate_fn
    )

    data_loader_mot17_test = DataLoader(
        mot17_dataset_test,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=mot17_dataset_test.standard_collate_fn
    )

    data_loader_crowd_human_test = DataLoader(
        crowd_human_dataset_test,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=crowd_human_dataset_test.standard_collate_fn
    )

    data_loader_city_persons_test = DataLoader(
        city_persons_dataset_test,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=city_persons_dataset_test.standard_collate_fn
    )

    data_loader_coco_persons_test = DataLoader(
        coco_persons_dataset_test,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=coco_persons_dataset_test.standard_collate_fn
    )

    # Creating model
    print("Creating model")
    model, backbone = get_model_detection(num_classes=1, model=args.model, pretrained=args.pretrained)

    # Putting model to device and setting eval mode
    model.to(device)
    model.train()

    # freeze the backbone parameters, if needed
    if backbone is not None and args.freeze_backbone:
        for param in backbone.parameters():
            param.requires_grad = False
        print('Backbone is freezed!')

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params,
                                    lr=args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay
                                    )
    elif args.optimizer == "adam":
        optimizer = torch.optim.Adam(
            params=params,
            lr=args.lr,
        )
    else:
        print("Optimizer not available")
        exit(1)

    # and a learning rate scheduler
    if args.lr_scheduler == "step_lr":
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                       step_size=args.lr_step_size,
                                                       gamma=args.lr_gamma
                                                       )
    elif args.lr_scheduler == "plateau":
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                                  mode='max',
                                                                  patience=args.lr_patience,
                                                                  verbose=True
                                                                  )
    else:
        print("L-Scheduler not available")
        exit(1)

    # Defining a warm-uo lr scheduler
    warmup_iters = min(1000, len(train_dataloader) - 1)
    warmup_factor = 1. / 1000
    warmup_lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    # Loading checkpoint
    start_epoch = 0
    train_step = -1
    best_viped_ap, best_mot19_ap, best_mot17_ap, best_crowdhuman_ap, best_citypersons_ap, best_cocopersons_ap \
        = 0, 0, 0, 0, 0, 0
    if args.resume:
        print("Resuming from checkpoint")
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        warmup_lr_scheduler.load_state_dict(checkpoint['warmup_lr_scheduler'])
        start_epoch = checkpoint['epoch']
        train_step = checkpoint['iteration']
        best_viped_ap = checkpoint['best_viped_ap']
        best_mot19_ap = checkpoint['best_mot19_ap']
        best_mot17_ap = checkpoint['best_mot17_ap']
        best_crowdhuman_ap = checkpoint['best_crowdhuman_ap']
        best_citypersons_ap = checkpoint['best_citypersons_ap']
        best_cocopersons_ap = checkpoint['best_cocopersons_ap']

    # Cross-check if the backbone has been really freezed
    if backbone is not None and args.freeze_backbone:
        for param in backbone.parameters():
            assert not param.requires_grad, "Backbone seems to be not freezed correctly!"

    # Train
    print("Start training")
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

            if (train_step % args.save_freq == 0 and train_step != 0) or \
               (args.pretrained and train_step < 5*args.save_freq and train_step % 200 == 0 and train_step != 0) \
                    or train_step == 100:
                # evaluate on the test datasets
                print("Validation viped Dataset")
                viped_coco_evaluator = evaluate(model, data_loader_viped_test, device=device, max_dets=args.max_dets)
                print("Validation mot19 Dataset")
                mot19_coco_evaluator = evaluate(model, data_loader_mot19_test, device=device, max_dets=args.max_dets)
                print("Validation mot17 Dataset")
                mot17_coco_evaluator = evaluate(model, data_loader_mot17_test, device=device, max_dets=args.max_dets)
                print("Validation crowdhuman Dataset")
                crowdhuman_coco_evaluator = evaluate(model, data_loader_crowd_human_test, device=device,
                                                     max_dets=args.max_dets)
                print("Validation citypersons Dataset")
                citypersons_coco_evaluator = evaluate(model, data_loader_city_persons_test, device=device,
                                                      max_dets=args.max_dets)
                print("Validation COCO Persons Dataset")
                cocopersons_coco_evaluator = evaluate(model, data_loader_coco_persons_test, device=device,
                                                      max_dets=args.max_dets)

                # save using tensorboard
                viped_ap, mot19_ap, mot17_ap, crowdhuman_ap, citypersons_ap, cocopersons_ap = \
                    None, None, None, None, None, None
                for iou_type, coco_eval in viped_coco_evaluator.coco_eval.items():
                    viped_ap = coco_eval.stats[1]
                for iou_type, coco_eval in mot19_coco_evaluator.coco_eval.items():
                    mot19_ap = coco_eval.stats[1]
                for iou_type, coco_eval in mot17_coco_evaluator.coco_eval.items():
                    mot17_ap = coco_eval.stats[1]
                for iou_type, coco_eval in crowdhuman_coco_evaluator.coco_eval.items():
                    crowdhuman_ap = coco_eval.stats[1]
                for iou_type, coco_eval in citypersons_coco_evaluator.coco_eval.items():
                    citypersons_ap = coco_eval.stats[1]
                for iou_type, coco_eval in cocopersons_coco_evaluator.coco_eval.items():
                    cocopersons_ap = coco_eval.stats[1]
                writer.add_scalar('COCO mAP Validation/ViPeD', viped_ap, train_step)
                writer.add_scalar('COCO mAP Validation/MOT19', mot19_ap, train_step)
                writer.add_scalar('COCO mAP Validation/MOT17', mot17_ap, train_step)
                writer.add_scalar('COCO mAP Validation/CrowdHuman', crowdhuman_ap, train_step)
                writer.add_scalar('COCO mAP Validation/CityPersons', citypersons_ap, train_step)
                writer.add_scalar('COCO mAP Validation/COCOPersons', cocopersons_ap, train_step)

                # Eventually saving best models
                if viped_ap > best_viped_ap:
                    best_viped_ap = viped_ap
                    save_checkpoint({
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'warmup_lr_scheduler':
                            warmup_lr_scheduler.state_dict() if warmup_lr_scheduler is not None else None,
                        'epoch': epoch,
                        'iteration': train_step,
                        'best_viped_ap': best_viped_ap,
                        'best_mot19_ap': best_mot19_ap,
                        'best_mot17_ap': best_mot17_ap,
                        'best_crowdhuman_ap': best_crowdhuman_ap,
                        'best_citypersons_ap': best_citypersons_ap,
                        'best_cocopersons_ap': best_cocopersons_ap,
                    }, writer.get_logdir(), best_model="viped")
                if mot19_ap > best_mot19_ap:
                    best_mot19_ap = mot19_ap
                    save_checkpoint({
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'warmup_lr_scheduler':
                            warmup_lr_scheduler.state_dict() if warmup_lr_scheduler is not None else None,
                        'epoch': epoch,
                        'iteration': train_step,
                        'best_viped_ap': best_viped_ap,
                        'best_mot19_ap': best_mot19_ap,
                        'best_mot17_ap': best_mot17_ap,
                        'best_crowdhuman_ap': best_crowdhuman_ap,
                        'best_citypersons_ap': best_citypersons_ap,
                        'best_cocopersons_ap': best_cocopersons_ap,
                    }, writer.get_logdir(), best_model="mot19")
                if mot17_ap > best_mot17_ap:
                    best_mot17_ap = mot17_ap
                    save_checkpoint({
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'warmup_lr_scheduler':
                            warmup_lr_scheduler.state_dict() if warmup_lr_scheduler is not None else None,
                        'epoch': epoch,
                        'iteration': train_step,
                        'best_viped_ap': best_viped_ap,
                        'best_mot19_ap': best_mot19_ap,
                        'best_mot17_ap': best_mot17_ap,
                        'best_crowdhuman_ap': best_crowdhuman_ap,
                        'best_citypersons_ap': best_citypersons_ap,
                        'best_cocopersons_ap': best_cocopersons_ap,
                    }, writer.get_logdir(), best_model="mot17")
                if crowdhuman_ap > best_crowdhuman_ap:
                    best_crowdhuman_ap = crowdhuman_ap
                    save_checkpoint({
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'warmup_lr_scheduler':
                            warmup_lr_scheduler.state_dict() if warmup_lr_scheduler is not None else None,
                        'epoch': epoch,
                        'iteration': train_step,
                        'best_viped_ap': best_viped_ap,
                        'best_mot19_ap': best_mot19_ap,
                        'best_mot17_ap': best_mot17_ap,
                        'best_crowdhuman_ap': best_crowdhuman_ap,
                        'best_citypersons_ap': best_citypersons_ap,
                        'best_cocopersons_ap': best_cocopersons_ap,
                    }, writer.get_logdir(), best_model="crowdhuman")
                if citypersons_ap > best_citypersons_ap:
                    best_citypersons_ap = citypersons_ap
                    save_checkpoint({
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'warmup_lr_scheduler':
                            warmup_lr_scheduler.state_dict() if warmup_lr_scheduler is not None else None,
                        'epoch': epoch,
                        'iteration': train_step,
                        'best_viped_ap': best_viped_ap,
                        'best_mot19_ap': best_mot19_ap,
                        'best_mot17_ap': best_mot17_ap,
                        'best_crowdhuman_ap': best_crowdhuman_ap,
                        'best_citypersons_ap': best_citypersons_ap,
                        'best_cocopersons_ap': best_cocopersons_ap,
                    }, writer.get_logdir(), best_model="citypersons")

                # Saving model
                save_checkpoint({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'warmup_lr_scheduler':
                        warmup_lr_scheduler.state_dict() if warmup_lr_scheduler is not None else None,
                    'epoch': epoch,
                    'iteration': train_step,
                    'best_viped_ap': best_viped_ap,
                    'best_mot19_ap': best_mot19_ap,
                    'best_mot17_ap': best_mot17_ap,
                    'best_crowdhuman_ap': best_crowdhuman_ap,
                    'best_citypersons_ap': best_citypersons_ap,
                    'best_cocopersons_ap': best_cocopersons_ap,
                }, writer.get_logdir())

                # Setting again to train mode
                model.train()

            lr_scheduler.step()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--dataset-path', default='', help='dataset root path')
    parser.add_argument('--model', default='fasterrcnn', help='model')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('-b', '--batch-size', default=6, type=int,
                        help='images per gpu, the total batch size is $NGPU x batch_size')
    parser.add_argument('--epochs', default=50, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 8)')
    parser.add_argument('--lr-scheduler', default="step_lr", help="LR Scheduler")
    parser.add_argument('--lr', default=0.005, type=float,
                        help='initial learning rate, 0.005 is the default value for training (fine tuning)')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=5e-4, type=float, metavar='W',
                        help='weight decay (default: 1e-4)', dest='weight_decay')
    parser.add_argument('--lr-step-size', default=20000, type=int, help='decrease lr every step-size epochs')
    parser.add_argument('--lr-patience', default=2, type=int,
                        help='patience before decreasing the lr every step-size epochs')
    parser.add_argument('--lr-gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')
    parser.add_argument('--print-freq', default=100, type=int, help='print frequency')
    parser.add_argument('--save-freq', default=1000, type=int, help='save frequency')
    parser.add_argument('--log-loss', default=10, type=int, help='save loss values using tensorboard')
    parser.add_argument('--output-dir', default='./output', help='path where to save output images')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--aspect-ratio-group-factor', default=0, type=int)
    parser.add_argument('--thresh', default=0.5, type=float, help="box score threshold")
    parser.add_argument('--max-dets', default=350, type=int, help="max num of detections per image")
    parser.add_argument('--train-on', default='viped', type=str, help="which dataset use for training")
    parser.add_argument('--tgt-images-in-batch', default=-1, type=int,
                        help="in case of mixed batches, how many target images in the batch")
    parser.add_argument('--pretrained', default=False, help="coco pretrained")
    parser.add_argument('--optimizer', default="sgd", help="Optimizer")
    parser.add_argument('--aug', default=None, help="Type of aug")
    parser.add_argument('--freeze-backbone', default=False, help="Freeze the backbone during train")

    args = parser.parse_args()

    print('Using pretrained model: {}'.format(args.pretrained))
    if args.pretrained:
        args.lr_step_size /= 4
    print('Learning rate step size: {}'.format(args.lr_step_size))

    main(args)
