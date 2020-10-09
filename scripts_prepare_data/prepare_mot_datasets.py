import os
import tqdm
import configparser
import numpy as np
from shutil import copyfile, move, rmtree
from random import shuffle


def main(args):
    print(args)

    # Training data
    mot_train_root_folder = os.path.join(args.mot_root_dir, "train")
    train_seq_folders = os.listdir(mot_train_root_folder)
    print(train_seq_folders)

    train_imgs_folder = os.path.join(args.mot_root_dir, "imgs", "train")
    if not os.path.exists(train_imgs_folder):
        os.makedirs(train_imgs_folder)
    train_bbs_folder = os.path.join(args.mot_root_dir, "bbs", "train")
    if not os.path.exists(train_bbs_folder):
        os.makedirs(train_bbs_folder)

    for seq in tqdm.tqdm(train_seq_folders):
        annots_filename = os.path.join(mot_train_root_folder, seq, 'gt', 'gt.txt')
        seq_info = os.path.join(mot_train_root_folder, seq, 'seqinfo.ini')
        config = configparser.ConfigParser()
        config.read(seq_info)
        seq_width, seq_height = int(config['Sequence']['imWidth']), int(config['Sequence']['imHeight'])
        print('{}x{}'.format(seq_width, seq_height))

        with open(annots_filename) as gt_file:
            v = np.genfromtxt(gt_file, delimiter=',')
        images = [i for i in os.listdir(os.path.join(mot_train_root_folder, seq, 'img1')) if i.endswith('.jpg')]

        for img in tqdm.tqdm(images):
            new_img_name = seq + "_" + img
            new_ann_name = new_img_name.rsplit(".", 1)[0] + ".txt"
            img_path = os.path.join(mot_train_root_folder, seq, 'img1', img)
            copyfile(img_path, os.path.join(train_imgs_folder, new_img_name))

            img_id_s = os.path.splitext(img)[0]
            try:
                img_id = int(img_id_s)
            except ValueError as e:
                print('Image name: {}'.format(img))
                exit(1)
            filename = os.path.join(train_bbs_folder, new_ann_name)
            if args.skip and os.path.isfile(filename):
                continue
            cur_detections = [d for d in v if d[0] == img_id and d[7] == 1]

            with open(filename, 'w') as f:
                for d in cur_detections:
                    x_tl = d[2] if d[2] >= 0 or not args.crop_bb else 0
                    y_tl = d[3] if d[3] >= 0 or not args.crop_bb else 0
                    w = d[4]
                    h = d[5]
                    w = w if x_tl + w <= seq_width or not args.crop_bb else seq_width - x_tl
                    h = h if y_tl + h <= seq_height or not args.crop_bb else seq_height - y_tl
                    x_center = x_tl + w / 2
                    y_center = y_tl + h / 2

                    # normalize
                    x_center /= seq_width
                    y_center /= seq_height
                    w /= seq_width
                    h /= seq_height

                    if x_center < 0 or y_center < 0 or x_center > 1 or y_center > 1:
                        continue

                    f.write('{} {} {} {} {}\n'.format(0, x_center, y_center, w, h))

    # Validation data
    val_imgs_folder = os.path.join(args.mot_root_dir, "imgs", "val")
    if not os.path.exists(val_imgs_folder):
        os.makedirs(val_imgs_folder)
    val_bbs_folder = os.path.join(args.mot_root_dir, "bbs", "val")
    if not os.path.exists(val_bbs_folder):
        os.makedirs(val_bbs_folder)
    train_imgs = os.listdir(train_imgs_folder)
    shuffle(train_imgs)
    num_total_imgs = len(train_imgs)
    num_val_imgs = int((num_total_imgs / 100) * 20)
    val_imgs = train_imgs[:num_val_imgs]
    for val_img in val_imgs:
        img_path = os.path.join(train_imgs_folder, val_img)
        move(img_path, os.path.join(val_imgs_folder, val_img))
        ann_name = val_img.rsplit(".", 1)[0] + ".txt"
        move(os.path.join(train_bbs_folder, ann_name), os.path.join(val_bbs_folder, ann_name))

    # Test data
    mot_test_root_folder = os.path.join(args.mot_root_dir, "test")
    test_seq_folders = os.listdir(mot_test_root_folder)
    print(test_seq_folders)

    test_imgs_folder = os.path.join(args.mot_root_dir, "imgs", "test")
    if not os.path.exists(test_imgs_folder):
        os.makedirs(test_imgs_folder)

    for seq in tqdm.tqdm(test_seq_folders):
        seq_info = os.path.join(mot_test_root_folder, seq, 'seqinfo.ini')
        config = configparser.ConfigParser()
        config.read(seq_info)
        seq_width, seq_height = int(config['Sequence']['imWidth']), int(config['Sequence']['imHeight'])
        print('{}x{}'.format(seq_width, seq_height))

        images = [i for i in os.listdir(os.path.join(mot_test_root_folder, seq, 'img1')) if i.endswith('.jpg')]

        for img in tqdm.tqdm(images):
            new_img_name = seq + "_" + img
            img_path = os.path.join(mot_test_root_folder, seq, 'img1', img)
            copyfile(img_path, os.path.join(test_imgs_folder, new_img_name))

    rmtree(mot_train_root_folder)
    rmtree(mot_test_root_folder)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('mot_root_dir', type=str, help='MOT root directory')
    parser.add_argument('--skip', action='store_true', default=False)
    parser.add_argument('--crop-bb', action='store_true', default=False)

    args = parser.parse_args()

    main(args)
