# This script check the annotations

import os
from PIL import Image
import tqdm


ROOT_DATA = "/mnt/Dati_SSD_1/datasets/pedestrian_detection"

SRC_IMGS = "{}/AllReal/imgs/train".format(ROOT_DATA)
SRC_BBS = "{}/AllReal/bbs/train".format(ROOT_DATA)
SRC_VAL_IMGS = "{}/AllReal/imgs/val".format(ROOT_DATA)
SRC_VAL_BBS = "{}/AllReal/bbs/val".format(ROOT_DATA)

DST_BBS = "{}/AllReal/bbs_parsed/train".format(ROOT_DATA)
DST_VAL_BBS = "{}/AllReal/bbs_parsed/val".format(ROOT_DATA)


CROP_BBS = False


if __name__ == "__main__":

    # Training
    for img_file in tqdm.tqdm(os.listdir(SRC_IMGS)):
        ann_file = img_file.rsplit(".", 1)[0] + ".txt"
        img_path = os.path.join(SRC_IMGS, img_file)
        img = Image.open(img_path)
        img_width, img_height = img.size
        ann_path = os.path.join(SRC_BBS, ann_file)
        with open(ann_path) as f:
            content = f.readlines()
        content = [x.strip() for x in content]

        for bb in content:
            bb = bb.split()
            if float(bb[3]) == 0.0 or float(bb[4]) == 0.0:
                print("{}: BB width or height is zero".format(ann_file))
                print("Removing ann and related image...")
                os.remove(ann_path)
                if os.path.isfile(ann_path.replace("bbs", "imgs").rsplit(".", 1)[0] + ".jpg"):
                    os.remove(ann_path.replace("bbs", "imgs").rsplit(".", 1)[0] + ".jpg")
                elif os.path.isfile(ann_path.replace("bbs", "imgs").rsplit(".", 1)[0] + ".png"):
                    os.remove(ann_path.replace("bbs", "imgs").rsplit(".", 1)[0] + ".png")
                else:
                    print("Img file not exists!")

        if CROP_BBS:
            new_anns = []
            for bb in content:
                bb = bb.split()
                x_tl = float((float(bb[1]) - (float(bb[3]) / 2)) * float(img_width))
                y_tl = float((float(bb[2]) - (float(bb[4]) / 2)) * float(img_height))
                bb_width = float(float(bb[3]) * float(img_width))
                bb_height = float(float(bb[4]) * float(img_height))
                x_br = x_tl + bb_width
                y_br = y_tl + bb_height

                if x_br <= x_tl or y_br <= y_tl:
                    print("Ann not correct")
                    print(img_file)
                    exit(1)

                if x_tl < 0.0:
                     #print("BB out of the image (left): {}".format(bb))
                     bb_width += abs(x_tl)
                     x_tl = 0.0

                if y_tl < 0.0:
                    #print("BB out of the image (top): {}".format(bb))
                    bb_height += abs(y_tl)
                    y_tl = 0.0

                if x_tl > float(img_width):
                    #print("BB out of the image (right): {}".format(bb))
                    bb_width -= abs(x_tl)
                    x_tl = float(img_width)

                if y_tl > float(img_height):
                    #print("BB out of the image (bottom): {}".format(bb))
                    bb_height -= abs(y_tl)
                    y_tl = float(img_height)

                x_center = (float(x_tl) + (float(bb_width) / 2)) / float(img_width)
                y_center = (float(y_tl) + (float(bb_height) / 2)) / float(img_height)
                bb_width = float(bb_width) / float(img_width)
                bb_height = float(bb_height) / float(img_height)

                new_anns.append(['0', x_center, y_center, bb_width, bb_height])

            with open(os.path.join(DST_BBS, ann_file), 'w') as file:
                for row in new_anns:
                    s = " ".join(map(str, row))
                    file.write(s + '\n')

    # Validation
    for img_file in tqdm.tqdm(os.listdir(SRC_VAL_IMGS)):
        ann_file = img_file.rsplit(".", 1)[0] + ".txt"
        img_path = os.path.join(SRC_VAL_IMGS, img_file)
        img = Image.open(img_path)
        img_width, img_height = img.size
        ann_path = os.path.join(SRC_VAL_BBS, ann_file)
        with open(ann_path) as f:
            content = f.readlines()
        content = [x.strip() for x in content]

        for bb in content:
            bb = bb.split()
            if float(bb[3]) == 0.0 or float(bb[4]) == 0.0:
                print("{}: BB width or height is zero".format(ann_file))
                print("Removing ann and related image...")
                os.remove(ann_path)
                if os.path.isfile(ann_path.replace("bbs", "imgs").rsplit(".", 1)[0] + ".jpg"):
                    os.remove(ann_path.replace("bbs", "imgs").rsplit(".", 1)[0] + ".jpg")
                elif os.path.isfile(ann_path.replace("bbs", "imgs").rsplit(".", 1)[0] + ".png"):
                    os.remove(ann_path.replace("bbs", "imgs").rsplit(".", 1)[0] + ".png")
                else:
                    print("Img file not exists!")

        if CROP_BBS:
            new_anns = []
            for bb in content:
                bb = bb.split()
                x_tl = float((float(bb[1]) - (float(bb[3]) / 2)) * float(img_width))
                y_tl = float((float(bb[2]) - (float(bb[4]) / 2)) * float(img_height))
                bb_width = float(float(bb[3]) * float(img_width))
                bb_height = float(float(bb[4]) * float(img_height))
                x_br = x_tl + bb_width
                y_br = y_tl + bb_height

                if x_br <= x_tl or y_br <= y_tl:
                    print("Ann not correct")
                    print(img_file)
                    exit(1)

                if x_tl < 0.0:
                     #print("BB out of the image (left): {}".format(bb))
                     bb_width += abs(x_tl)
                     x_tl = 0.0

                if y_tl < 0.0:
                    #print("BB out of the image (top): {}".format(bb))
                    bb_height += abs(y_tl)
                    y_tl = 0.0

                if x_tl > float(img_width):
                    #print("BB out of the image (right): {}".format(bb))
                    bb_width -= abs(x_tl)
                    x_tl = float(img_width)

                if y_tl > float(img_height):
                    #print("BB out of the image (bottom): {}".format(bb))
                    bb_height -= abs(y_tl)
                    y_tl = float(img_height)

                x_center = (float(x_tl) + (float(bb_width) / 2)) / float(img_width)
                y_center = (float(y_tl) + (float(bb_height) / 2)) / float(img_height)
                bb_width = float(bb_width) / float(img_width)
                bb_height = float(bb_height) / float(img_height)

                new_anns.append(['0', x_center, y_center, bb_width, bb_height])

            with open(os.path.join(DST_VAL_BBS, ann_file), 'w') as file:
                for row in new_anns:
                    s = " ".join(map(str, row))
                    file.write(s + '\n')
