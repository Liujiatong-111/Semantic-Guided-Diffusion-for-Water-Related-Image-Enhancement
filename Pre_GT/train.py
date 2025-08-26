import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataloder.data_loder import llvip
from models.common import clamp
from models.model import GT_model
from pytorch_msssim import ssim



def init_seeds(seed=0):
    # Initialize random number generator (RNG) seeds https://pytorch.org/docs/stable/notes/randomness.html
    # cudnn seed 0 settings are slower and more reproducible, else faster and less reproducible
    import torch.backends.cudnn as cudnn
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    cudnn.benchmark, cudnn.deterministic = (False, True) if seed == 0 else (True, False)


if __name__ == '__main__':
    init_seeds(42)
    datasets = '/home/groupyun/桌面/sdd_new/水下图像diff/datasets'
    save_path = 'runs/'
    batch_size = 16
    num_works = 8
    lr = 0.001
    Epoch = 30

    # 数据集加载
    train_dataset = llvip(datasets)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_works, pin_memory=True)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 创建模型
    model = GT_model(in_channel=6).cuda()

    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 初始化自动混合精度工具
    scaler = torch.cuda.amp.GradScaler()  # 用于鉴别器的梯度缩放

    model.train()

    for epoch in range(Epoch):
        train_tqdm = tqdm(train_loader, total=len(train_loader), ascii=True)
        for img1, img2, img1_map, img2_map, name in train_tqdm:
            optimizer.zero_grad()

            img1 = img1.cuda()
            img2 = img2.cuda()

            img1_map = img1_map.cuda()
            img2_map = img2_map.cuda()

            with torch.cuda.amp.autocast():  # 开启混合精度
                f = model(torch.cat([img1, img2], dim=1))
                f = clamp(f)
                # ---损失函数---#
                stacked = torch.stack((img1_map, img2_map), dim=1)
                weights = F.softmax(stacked, dim=1)
                weight1, weight2 = weights[:, 0], weights[:, 1]
                loss1 = F.l1_loss(f, weight1 * img1) + F.l1_loss(f, weight2 * img2)
                loss2 = 2 - ssim(f, img1) - ssim(f, img2)
                loss = loss1 + loss2

            scaler.scale(loss).backward()  # 使用梯度缩放
            scaler.step(optimizer)  # 更新生成器的参数
            scaler.update()  # 更新缩放因子

            # ---显示loss
            train_tqdm.set_postfix(epoch=epoch,
                                   loss=loss.item())

        # 保存训练模型
        torch.save(model.state_dict(), fr'{save_path}/{epoch}.pth')

