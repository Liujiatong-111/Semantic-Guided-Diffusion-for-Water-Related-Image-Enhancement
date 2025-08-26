import os
from PIL import Image


def resize_images_in_folder(folder_path, target_size=(256, 256)):
    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        # 获取文件的完整路径
        file_path = os.path.join(folder_path, filename)

        # 检查是否为图像文件
        if os.path.isfile(file_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            try:
                # 打开图像
                img = Image.open(file_path)

                # 调整图像大小
                img_resized = img.resize(target_size)

                # 保存调整后的图像（你可以选择覆盖原图像或者保存为新文件）
                img_resized.save(file_path)  # 覆盖原图像
                # 或者使用：img_resized.save(os.path.join(folder_path, "resized_" + filename))  # 保存为新文件

                print(f"已处理: {filename}")
            except Exception as e:
                print(f"处理 {filename} 时出错: {e}")


# 使用示例
folder_path = r'G:\2024F_10_16\扩散水下图像增强\UW-DIFFPHYS'  # 替换为你的文件夹路径
resize_images_in_folder(folder_path)
