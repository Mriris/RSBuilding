import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def visualize_difference(image1_path, image2_path):
    # 加载图像
    image1 = np.array(Image.open(image1_path))
    image2 = np.array(Image.open(image2_path))

    # 计算差异图
    diff = np.abs(image1 - image2)

    # 显示图像
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.title("图像1")
    plt.imshow(image1)
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("图像2")
    plt.imshow(image2)
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("差异图")
    plt.imshow(diff, cmap='hot')  # 使用热力图显示差异
    plt.axis('off')

    plt.show()

# 示例用法
image1_path = "/home/iris/PycharmProjects/MambaCD/results/mua2/MambaBCD_Tiny/change_map/0003.png"  # 第一组图像路径
image2_path = "/home/iris/PycharmProjects/MambaCD/results/mua2/MambaBCD_Tiny/change_map/test_1.png"  # 第二组图像路径
visualize_difference(image1_path, image2_path)
