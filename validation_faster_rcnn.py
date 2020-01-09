import os
import tqdm

import torch
from torch.utils.data import DataLoader

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from references.detection import utils
from references.detection.engine import evaluate as evaluate_coco

from utils import transforms as custom_T
from datasets.custom_yolo_annotated_dataset import CustomYoloAnnotatedDataset
import datetime

BASEDIR = 'validations'

DATASETS = {
    "viped": "/mnt/pedestrian_detection/ViPeD",
    "mot19": "/mnt/pedestrian_detection/MOT19Det",
    "mot17": "/mnt/pedestrian_detection/MOT17Det",
    "mot19_vs": "/mnt/pedestrian_detection/MOT19Det_Vertical_Split",
    "mot17_vs": "/mnt/pedestrian_detection/MOT17Det_Vertical_Split",
    "crowd_human": "/mnt/pedestrian_detection/crowd_human",
    "city_persons": "/mnt/pedestrian_detection/city_persons",
    "COCO_persons": "/mnt/pedestrian_detection/COCO_Persons",
}

# RESULT_FILE = os.path.join("results", "faster_rcnn_coco_validation_thre05_maxDets350.txt")
# with open(RESULT_FILE, 'w'):
#     os.utime(RESULT_FILE, None)


def get_dataset(name, transform, percentage=None, img_size=None, split="train"):
    if name in DATASETS:
        dataset_root_path = DATASETS[name]
        dataset = CustomYoloAnnotatedDataset(dataset_root_path, transform, dataset_name=name, img_size=img_size,
                                             percentage=percentage, split=split)
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
    # transforms.append(custom_T.FasterRCNNResizer())

    return custom_T.Compose(transforms)


def evaluate(args, model, dataset, experiment_base):
    dset = dataset()
    data_loader = DataLoader(
        dset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=dset.standard_collate_fn
    )
    if args.evaluation_apis == 'coco':
        filename = os.path.join(experiment_base, '{}.txt'.format(data_loader.dataset.dataset_name))
        evaluate_coco(model, data_loader, device=args.device, categories=[1], save_on_file=filename,
                      dataset_name=dset.dataset_name, max_dets=args.max_dets)
    elif args.evaluation_apis == 'mot':
        dirname = os.path.join(experiment_base, data_loader.dataset.dataset_name)
        os.makedirs(dirname)
        evaluate_mot(model, data_loader, device=args.device, categories=[1], save_dir=dirname,
                     dataset_name=dset.dataset_name)


def main(args):
    utils.init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)

    # Creating model
    print("Creating model")
    num_classes = 2
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False,
                                                                 box_detections_per_img=args.max_dets,
                                                                 box_score_thresh=args.thresh)
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # generate a file with specific validation name
    checkpoint_dir = os.path.dirname(args.resume).rsplit("/", 1)[1]
    experiment_base = '{}_{}-evaluation_thresh{}_validateon{}___{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M"),
                                                         args.evaluation_apis, args.thresh, args.validate_on, checkpoint_dir)
    experiment_base = os.path.join(BASEDIR, experiment_base)
    if not os.path.exists(experiment_base):
        os.makedirs(experiment_base)

    # Loading saved checkpoint
    if args.resume:
        loaded_weights = torch.load(args.resume, map_location=device)['model']
        model.load_state_dict(loaded_weights)
        print('Loaded checkpoint from {}'.format(args.resume))
    else:
        raise ValueError('RESUME argument must be set!')

    # Putting model to device and setting eval mode
    model.to(device)
    model.eval()

    if args.evaluation_apis == "coco":
        split = "val"
    else:
        split = "test"

    test_dataset_dict = {
        'viped': lambda: get_dataset("viped", get_transform(train=False, aug=args.aug), split=split),
        'mot19': lambda: get_dataset("mot19", get_transform(train=False), split=split),
        'mot17': lambda: get_dataset("mot17", get_transform(train=False), split=split),
        'crowd_human': lambda: get_dataset("crowd_human", get_transform(train=False), split=split),
        'city_persons': lambda: get_dataset("city_persons", get_transform(train=False), split=split),
        'COCO_persons': lambda: get_dataset("COCO_persons", get_transform(train=False), split=split)
    }

    # Evaluate (only on category 1 = person)
    if args.validate_on == "all":
        for d_name, dataset in tqdm.tqdm(test_dataset_dict.items()):
            if args.evaluation_apis == 'mot' and (d_name != 'mot17' and d_name != 'mot19'):
                # as of now, if using mot evaluator you can only evaluate mot datasets
                continue

            print("Validation {} Dataset".format(d_name))
            evaluate(args, model, dataset, experiment_base)
    else:
        evaluate(args, model, test_dataset_dict[args.validate_on], experiment_base)

    print('DONE!')


def evaluate_mot(model, data_loader, device, categories=None, save_dir=None, dataset_name=None):
    detections_per_subseq = {}

    for batch_i, (imgs, _) in enumerate(tqdm.tqdm(data_loader,
                                                  desc="Accumulating detections for MOT evaluation...")):
        # Get detections
        imgs = list(img.to(device) for img in imgs)
        with torch.no_grad():
            detections = model(imgs)

        # Get image filenames for this batch
        img_filenames = [data_loader.dataset.imgs_id_path[batch_i*len(imgs) + i]
                         for i in range(len(imgs))]

        # iterate over images on this batch
        for det, img_filename in zip(detections, img_filenames):
            subseq, id = os.path.basename(img_filename).rsplit('.', 1)[0].split('_')
            # move detections to cpu
            det = {k: v.cpu() for k, v in det.items()}
            # iterate over single detections in this image
            for score, box, label in zip(det['scores'], det['boxes'], det['labels']):
                if subseq not in detections_per_subseq:
                    detections_per_subseq[subseq] = []
                row = [id, label, score, box]
                detections_per_subseq[subseq].append(row)

        # if batch_i == 5:
        #     break

    print('Writing on MOT txt files in {}'.format(save_dir))
    for k, objs in detections_per_subseq.items():
        with open('{}.txt'.format(os.path.join(save_dir, k)), 'w') as outf:
            for det in objs:
                if det[1] == 1:  # it is a person
                    x = det[3][0]
                    y = det[3][1]
                    w = det[3][2] - det[3][0]
                    h = det[3][3] - det[3][1]
                    line = '{}, {}, {}, {}, {}, {}, {}'.format(
                        int(det[0]), -1, x, y, w, h, det[2])
                    outf.write('{}\n'.format(line))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('resume', default='', help='resume from checkpoint')

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
    parser.add_argument('--aspect-ratio-group-factor', default=0, type=int)
    parser.add_argument('--thresh', default=0.5, type=float, help="box score threshold")
    parser.add_argument('--max-dets', default=350, type=int, help="max num of detections per image")
    parser.add_argument('--tgt-images-in-batch', default=-1, type=int,
                        help="in case of mixed batches, how many target images in the batch")
    parser.add_argument('--optimizer', default="sgd", help="Optimizer")
    parser.add_argument('--aug', default=None, help="Type of aug")
    parser.add_argument('--freeze-backbone', default=False, help="Freeze the backbone during train")
    parser.add_argument('--evaluation-apis', default='mot', help="Which validator you want to use")
    parser.add_argument('--validate-on', default='all', help="Which dataset you want to validate on")

    args = parser.parse_args()

    if args.output_dir:
        utils.mkdir(args.output_dir)

    main(args)
