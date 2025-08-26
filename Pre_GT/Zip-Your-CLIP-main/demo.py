import os
import numpy as np
import torch
from PIL import Image
import cv2
from zip_utils import *
from skimage import measure
import numpy as np
from PIL import Image
# 读取文本文件并将每一行作为一个文本特征
def read_texts_from_file(file_path):
    with open(file_path, 'r') as f:
        # 读取每一行并去掉两端的空白符
        texts = [line.strip() for line in f.readlines()]
    return texts


def main(img_path1, text_path1, save_path1, model):

    # 图像路径
    img_path = img_path1
    # 定义文本特征
    # 示例：读取文本
    file_path = text_path1
    all_texts = read_texts_from_file(file_path)

    # 使用 PIL 打开图像
    I = Image.open(img_path).convert('RGB')

    # 使用 OpenCV 读取图像并转换为 RGB
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 将图像转换为 Tensor 并传输到 GPU
    tensor = H2_ToTensor(I).unsqueeze(0)  # 假设 H2_ToTensor 是一个自定义函数
    tensor = tensor.cuda()



    # 提取文本特征
    with torch.no_grad():
        text_features = clip.encode_text_with_prompt_ensemble(model, all_texts, 'cuda')

    # 提取冗余文本特征
    all_texts = ['']
    with torch.no_grad():
        redundant_feats = clip.encode_text_with_prompt_ensemble(model, all_texts, 'cuda')

    # 提取图像特征
    with torch.no_grad():
        image_features0 = model.encode_image(tensor)
        image_features0 = image_features0 / image_features0.norm(dim=1, keepdim=True)

        # 计算相似度
        similarity0 = clip.clip_feature_surgery(image_features0, text_features, redundant_feats)
        similarity_map0 = clip.get_similarity_map(similarity0[:, 1:, :], image.shape[:2]).squeeze(0)
        similarity_map0 = similarity_map0.cpu().numpy()



    # 假设 similarity_map0 是计算得到的相似度图
    similarity_map0 = similarity_map0 * 255  # 转换为 [0, 255] 范围
    similarity_map0 = np.uint8(similarity_map0)  # 转换为 uint8 类型

    # 检查 similarity_map0 的维度和类型
    print(similarity_map0.shape, similarity_map0.dtype)

    # 如果 similarity_map0 是单通道（灰度图）
    if len(similarity_map0.shape) == 2:
        similarity_map_img = Image.fromarray(similarity_map0)
    elif len(similarity_map0.shape) == 3 and similarity_map0.shape[2] == 1:
        similarity_map0 = similarity_map0.squeeze(axis=2)  # 去掉多余的通道维度
        similarity_map_img = Image.fromarray(similarity_map0)
    else:
        raise ValueError("Unexpected image dimensions")

    # 保存图像
    similarity_map_img.save(save_path1)

