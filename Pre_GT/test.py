"""测试融合网络"""
import argparse
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from dataloder.data_loder import llvip
from models.model import GT_model
from models.common import clamp
import time
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'





if __name__ == '__main__':
    num_works = 1
    test_data = '/home/groupyun/桌面/sdd_new/水下图像diff/datasets'
    print(test_data)
    fusion_result_path = test_data + '/gt'

    if not os.path.exists(fusion_result_path):
        os.makedirs(fusion_result_path)


    test_dataset = llvip(test_data)
    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False,
        num_workers=num_works, pin_memory=True)

    if not os.path.exists(fusion_result_path):
        os.makedirs(fusion_result_path)

    #######加载模型
    model = GT_model(in_channel=6).cuda()

    model.load_state_dict(torch.load('runs/29.pth'))

    model.eval()



    ##########加载数据
    test_tqdm = tqdm(test_loader, total=len(test_loader))
    with torch.no_grad():
        for img1, img2, img1_map, img2_map, name in test_tqdm:
            img1 = img1.cuda()
            img2 = img2.cuda()
            ###########转为rgb
            f = model(torch.cat([img1, img2], dim=1))
            fused = clamp(f)
            rgb_fused_image = transforms.ToPILImage()(f[0])
            rgb_fused_image.save(f'{fusion_result_path}/{name[0]}')
