import os
from demo import *
# 加载 CLIP 模型
model, _ = clip.load("CS-ViT-B/16")
model.eval().cuda()
def get_image_filenames(folder_path, extensions=['.png']):
    """
    获取指定文件夹下的所有图像文件名。

    :param folder_path: 图像文件夹路径
    :param extensions: 允许的图像文件扩展名列表
    :return: 图像文件名列表
    """
    # 获取文件夹下所有文件
    filenames = os.listdir(folder_path)

    # 过滤出图像文件
    image_filenames = [filename for filename in filenames if any(filename.lower().endswith(ext) for ext in extensions)]

    return image_filenames

# 示例用法
folder_path = "/home/groupyun/桌面/sdd_new/水下图像diff/datasets/osmosis"  # 请替换为实际文件夹路径
image_files = get_image_filenames(folder_path)

# 打印所有图像文件名
for image in image_files:
    image_name = image
    text_name = image_name.replace('.png','.txt')
    image_path = os.path.join(folder_path, image_name)
    text_path = os.path.join('/home/groupyun/桌面/sdd_new/水下图像diff/datasets/text', text_name)
    save_path = os.path.join('/home/groupyun/桌面/sdd_new/水下图像diff/datasets/osmosis_map', image_name)
    main(img_path1=image_path, text_path1=text_path, save_path1=save_path, model=model)

