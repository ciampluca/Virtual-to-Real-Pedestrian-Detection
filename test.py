import os
import tqdm
import yaml

import torch
from torch.utils.data import DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from references.detection import utils
from references.detection.engine import evaluate as evaluate_coco

from utils import transforms as custom_T
from datasets.custom_yolo_annotated_dataset import CustomYoloAnnotatedDataset
from models.faster_rcnn import fasterrcnn_resnet50_fpn, fasterrcnn_resnet101_fpn


def get_transform():
    transforms = []

    transforms.append(custom_T.ToTensor())
    transforms.append(custom_T.FasterRCNNResizer())

    return custom_T.Compose(transforms)


def evaluate(cfg, model, dataloader, dataset_name, split, args):
    # Retrieving model name
    model_name = args.load_model

    # Retrieving evaluation apis
    eval_api = cfg["evaluation_api"]
    assert eval_api == "coco" or eval_api == "mot", "Not valid evaluation apis"
    if eval_api == "coco" and split == "test":
        print("coco eval api not yet implemented for test split. Please use val split or mot eval api.")
        exit(1)

    # Retrieving folder result
    results_folder = cfg['results_folder']
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    if eval_api == 'coco':
        filename = os.path.join(results_folder, '{}_{}_{}.txt'.
                                format(dataset_name, ''.join(model_name.split("_")), "coco-eval"))
        evaluate_coco(model, dataloader, device=next(model.parameters()).device, categories=[1], save_on_file=filename,
                      max_dets=cfg['max_dets_per_image'])
    elif eval_api == 'mot':
        if 'joint' in dataset_name:
            pass
        results_folder = os.path.join(results_folder, "{}_{}_{}".
                                      format(dataset_name, ''.join(model_name.split("_")), "mot-eval"))
        if not os.path.exists(results_folder):
            os.makedirs(results_folder)
        evaluate_mot(model, dataloader, device=next(model.parameters()).device,
                     dataset_name=dataset_name, save_dir=results_folder)


def evaluate_mot(model, data_loader, device, dataset_name=None, save_dir=None):
    detections_per_subseq = {}

    for batch_i, (imgs, _) in enumerate(tqdm.tqdm(data_loader,
                                                  desc="Accumulating detections for MOT evaluation...")):
        # Get detections
        imgs = list(img.to(device) for img in imgs)
        with torch.no_grad():
            detections = model(imgs)

        # Get image file names for this batch
        img_filenames = [data_loader.dataset.images[dataset_name][batch_i*len(imgs) + i] for i in range(len(imgs))]

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

    # Opening YAML cfg config file
    with open(args.cfg_file, 'r') as stream:
        try:
            cfg_file = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    # Retrieving cfg
    test_cfg = cfg_file['test']
    model_cfg = cfg_file['model']
    data_cfg = cfg_file['dataset']

    # Setting device
    device = torch.device(model_cfg['device'])

    # Retrieving pretrained model
    available_pretrained_models = test_cfg['pretrained_models']
    pretrained_model_name = args.load_model
    assert pretrained_model_name in available_pretrained_models.keys(), \
        "Pretrained model {} not available".format(pretrained_model_name)
    checkpoint_path = available_pretrained_models[pretrained_model_name]

    # Creating model
    print("Creating model")
    if "50" in pretrained_model_name:
        model = fasterrcnn_resnet50_fpn(
            pretrained=False,
            pretrained_backbone=False,
            box_detections_per_img=model_cfg["max_dets_per_image"],
            box_nms_thresh=model_cfg["nms"],
            model_dir=model_cfg["cache_folder"],
        )
    else:
        model = fasterrcnn_resnet101_fpn(
            pretrained=False,
            pretrained_backbone=False,
            box_detections_per_img=model_cfg["max_dets_per_image"],
            box_score_thresh=cfg["det_thresh"],
            box_nms_thresh=model_cfg["nms"],
            model_dir=model_cfg["cache_folder"],
        )

    # Loading weights
    if not "coco" in pretrained_model_name:
        num_classes = 1 + 1  # num classes + background
        # Getting number of input features for the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        # Replacing the pre-trained head with a new one
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    if checkpoint_path.startswith('http://') or checkpoint_path.startswith('https://'):
        checkpoint = torch.hub.load_state_dict_from_url(
            checkpoint_path, map_location='cpu', model_dir=model_cfg["cache_folder"])
    else:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

    if 'model' in checkpoint.keys():
        checkpoint = checkpoint['model']
    model.load_state_dict(checkpoint)

    # Putting model to device and setting eval mode
    model.to(device)
    model.eval()

    # Retrieving phase and some data parameters
    phase = test_cfg['phase']
    assert phase == "test" or phase == "val", "Not valid phase"
    data_root = data_cfg['root']
    datasets_names = data_cfg[phase]

    # Creating dataset(s) and dataloader(s)
    for dataset_name, dataset_cfg in datasets_names.items():
        # Creating dataset
        dataset = CustomYoloAnnotatedDataset(
            data_root,
            {dataset_name: dataset_cfg},
            transforms=get_transform(),
            phase=phase
        )
        dataloader = DataLoader(
            dataset,
            batch_size=test_cfg['batch_size'],
            shuffle=False,
            num_workers=test_cfg['num_workers'],
            collate_fn=dataset.standard_collate_fn
        )

        # Evaluate
        evaluate(test_cfg, model, dataloader, dataset_name, split=dataset_cfg.rsplit(".", 1)[1], args=args)

    print('DONE!')


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--cfg-file', default='./cfg/config.yaml', help="YAML config file path")
    parser.add_argument('--load_model', default='coco_resnet_50',
                        help='Load a model. Possible values are contained in the cfg file (test section) ')

    args = parser.parse_args()

    main(args)
