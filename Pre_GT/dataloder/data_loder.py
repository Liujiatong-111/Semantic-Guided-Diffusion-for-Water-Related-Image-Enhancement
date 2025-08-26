import os

import numpy as np
from PIL import Image
from torch.utils import data
from torchvision import transforms

to_tensor = transforms.Compose([transforms.ToTensor()])



class llvip(data.Dataset):
    def __init__(self, data_dir, transform=to_tensor):
        super().__init__()
        dirname = os.listdir(data_dir)  # 获得TNO数据集的子目录
        for sub_dir in dirname:
            temp_path = os.path.join(data_dir, sub_dir)
            if sub_dir == 'osmosis':
                self.img1 = temp_path  # 获得红外路径
            elif sub_dir == 'UW-DIFFPHYS':
                self.img2 = temp_path  # 获得可见光路径

        self.name_list = os.listdir(self.img1)  # 获得子目录下的图片的名称
        self.transform = transform

    def __getitem__(self, index):
        name = self.name_list[index]  # 获得当前图片的名称

        img1 = Image.open(os.path.join(self.img1, name))
        img2 = Image.open(os.path.join(self.img2, name))
        img1_map = Image.open(os.path.join(self.img1 + '_map', name))
        img2_map = Image.open(os.path.join(self.img2 + '_map', name))

        img1 = self.transform(img1)
        img2 = self.transform(img2)
        img1_map = self.transform(img1_map)
        img2_map = self.transform(img2_map)

        return img1, img2, img1_map, img2_map, name

    def __len__(self):
        return len(self.name_list)


