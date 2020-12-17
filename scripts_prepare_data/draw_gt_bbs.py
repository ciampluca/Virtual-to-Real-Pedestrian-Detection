import os
from PIL import Image, ImageDraw

ROOT_IMG = "/mnt/Dati_SSD_1/datasets/pedestrian_detection/AllReal/imgs/train"

IMG = "MOT17-11_000171.jpg"

IMG_PATH = "{}/{}".format(ROOT_IMG, IMG)


if __name__ == "__main__":
    img = Image.open(IMG_PATH)
    img_width, img_height = img.size
    img_draw = ImageDraw.Draw(img)
    ann_path = os.path.join(ROOT_IMG.replace("imgs", "bbs_parsed"), IMG.rsplit(".", 1)[0] + ".txt")
    with open(ann_path) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    for bb in content:
        bb = bb.split()
        x_tl = float((float(bb[1]) - (float(bb[3]) / 2)) * float(img_width))
        y_tl = float((float(bb[2]) - (float(bb[4]) / 2)) * float(img_height))
        bb_width = float(float(bb[3]) * float(img_width))
        bb_height = float(float(bb[4]) * float(img_height))
        x_br = x_tl + bb_width
        y_br = y_tl + bb_height

        img_draw.rectangle(((x_tl, y_tl), (x_br, y_br)), outline="red", width=3)

    img.show()
