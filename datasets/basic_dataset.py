import glob
from PIL import Image

from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch.nn.functional as F


def resize_image(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image


class BasicDataset(Dataset):

    def __init__(self, folder_path, img_size=None):
        self.files = sorted(glob.glob("%s/*.*" % folder_path))
        self.img_size = img_size

    def __getitem__(self, index):
        img_path = self.files[index % len(self.files)]
        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(Image.open(img_path))
        # Pad to square and resize (useful with YOLO)
        if self.img_size:
            #img, _ = pad_to_square(img, 0)
            img = resize_image(img, self.img_size)

        return img_path, img

    def __len__(self):
        return len(self.files)
