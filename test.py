import os
import tqdm
import datetime

import torch
from torch.utils.data import DataLoader

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from references.detection import utils
from references.detection.engine import evaluate as evaluate_coco

from utils import transforms as custom_T
from datasets.custom_yolo_annotated_dataset import CustomYoloAnnotatedDataset


PRETRAINED_MODELS = {
    "viped": "./checkpoints/model_pretrained_viped.pth",
    "viped_mot17": "./checkpoints/model_pretrained_viped_mot17.pth",
    "viped_mot20": "./checkpoints/model_pretrained_viped_mot20.pth",
    "viped_mot17_mb": "./checkpoints/model_pretrained_viped_mot17_mb.pth",
    "viped_mot20_mb": "./checkpoints/model_pretrained_viped_mot20_mb.pth",
    "viped_cocopersons_mb": "./checkpoints/model_pretrained_viped_cocopersons_mb.pth",
}

DATASETS = {
    "viped": "./data/viped",
    "MOT17Det": "./data/MOT17Det",
    "MOT20Det": "./data/MOT20Det",
    "COCOPersons": "./data/COCOPersons",
}


def get_dataset(name, transform, percentage=None, split="train"):
    if name in DATASETS:
        dataset_root_path = DATASETS[name]
        dataset = CustomYoloAnnotatedDataset(dataset_root_path, transform, dataset_name=name, percentage=percentage,
                                             split=split)
    else:
        raise ValueError("Non existing dataset")

    return dataset


def get_transform():
    transforms = []

    transforms.append(custom_T.ToTensor())

    return custom_T.Compose(transforms)


def evaluate(args, model, dataset, eval_apis, folder_name):
    # Creating data loader
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=dataset.standard_collate_fn
    )

    if eval_apis == 'coco':
        filename = os.path.join(folder_name, '{}.txt'.format(data_loader.dataset.dataset_name))
        evaluate_coco(model, data_loader, device=args.device, categories=[1], save_on_file=filename,
                      dataset_name=dataset.dataset_name, max_dets=args.max_dets)
    elif eval_apis == 'mot':
        dirname = os.path.join(folder_name, data_loader.dataset.dataset_name)
        os.makedirs(dirname)
        evaluate_mot(model, data_loader, device=args.device, save_dir=dirname)


def evaluate_mot(model, data_loader, device, save_dir=None):
    detections_per_subseq = {}

    for batch_i, (imgs, _) in enumerate(tqdm.tqdm(data_loader,
                                                  desc="Accumulating detections for MOT evaluation...")):
        # Get detections
        imgs = list(img.to(device) for img in imgs)
        with torch.no_grad():
            detections = model(imgs)

        # Get image file names for this batch
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


def main(args):
    utils.init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)

    # Retrieving checkpoint
    checkpoint_path = PRETRAINED_MODELS.get(args.resume, args.resume)
    assert checkpoint_path.endswith(".pth") or args.resume == "coco_original", "Not valid checkpoint"

    # Creating model
    print("Creating model")
    if checkpoint_path.endswith(".pth"):
        num_classes = 2
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False,
                                                                     box_detections_per_img=args.max_dets,
                                                                     box_score_thresh=args.thresh)

        # get number of input features for the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        # Loading saved checkpoint
        loaded_weights = torch.load(checkpoint_path, map_location=device)['model']
        model.load_state_dict(loaded_weights)
        print('Loaded checkpoint from {}'.format(args.resume))
    elif args.resume == "coco_original":
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True,
                                                                     box_detections_per_img=args.max_dets,
                                                                     box_score_thresh=args.thresh)

    # Putting model to device and setting eval mode
    model.to(device)
    model.eval()

    # Defining dataset split and evaluation apis
    split = args.dataset_split
    eval_apis = args.evaluation_apis
    assert split == "test" or split == "val", "Not valid dataset split"
    assert eval_apis == "coco" or eval_apis == "mot", "Not valid evaluation apis"
    if eval_apis == "coco" and split == "test":
        print("COCO eval apis not yet implemented for test subset. Please use val subset or MOT eval apis.")
        exit(1)

    # Creating dataset
    dataset = get_dataset(args.dataset_name, get_transform(), split=split)

    # Defining folder name that will contain results
    base_name = "{}_{}-evaluation_thresh{}_on{}".\
        format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M"), eval_apis, args.thresh, args.dataset_name)
    res_folder_name = os.path.join("./test_results", base_name)

    if not os.path.exists(res_folder_name):
        os.makedirs(res_folder_name)

    # Evaluate
    evaluate(args, model, dataset, eval_apis, res_folder_name)

    print('DONE!')


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume', default='viped',
                        help='Resume from checkpoint. Possible values are viped (default), viped_mot17, viped_mot20, '
                             'viped_mot17_mb, viped_mot20_mb, viped_cocopersons_mb. '
                             'Otherwise give the .pth path of your custom model. '
                             'Finally, there is also the possibility to insert the value coco_original. In this case, '
                             'the original model pre-trained on COCO (80 classes) is loaded. ')
    parser.add_argument('--dataset-name', default='MOT17Det',
                        help='Dataset name. Possible values are MOT17Det (default), MOT20Det')
    parser.add_argument('--device', default='cuda', help='Device. Default is cuda')
    parser.add_argument('-b', '--batch-size', default=1, type=int,
                        help='Images per gpu, the total batch size is $NGPU x batch_size')
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='Number of data loading workers (default: 8)')
    parser.add_argument('--thresh', default=0.05, type=float, help="Box score threshold. Default 0.05")
    parser.add_argument('--max-dets', default=350, type=int, help="Max num of detections per image")
    parser.add_argument('--evaluation-apis', default='coco',
                        help="Which validator you want to use. Possible values are mot (default) and coco")
    parser.add_argument('--dataset-split', default='test',
                        help="Which dataset split you want to use. Possible values are test (default) and val. "
                             "GT are available only for the val subset")

    args = parser.parse_args()

    main(args)
