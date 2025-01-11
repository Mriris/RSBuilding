from PIL import Image
import numpy as np

def compare_images(image1_path, image2_path):
    """
    比较两幅图像的属性和内容。

    参数:
    - image1_path (str): 第一组图像路径（参考图像）。
    - image2_path (str): 第二组图像路径（待验证图像）。

    输出:
    打印两幅图像的分辨率、通道数、像素值范围、以及像素差异统计。
    """
    # 加载图像
    image1 = Image.open(image1_path)
    image2 = Image.open(image2_path)

    # 检查分辨率
    print(f"图像1分辨率: {image1.size}, 图像2分辨率: {image2.size}")

    # 检查通道模式
    print(f"图像1模式: {image1.mode}, 图像2模式: {image2.mode}")

    # 转换为 numpy 数组
    array1 = np.array(image1)
    array2 = np.array(image2)

    # 检查像素值范围
    print(f"图像1像素值范围: {array1.min()} - {array1.max()}")
    print(f"图像2像素值范围: {array2.min()} - {array2.max()}")

    # 确保尺寸一致
    if array1.shape != array2.shape:
        print("图像尺寸不同，无法直接比较内容！")
        return

    # 计算像素差异
    diff = np.abs(array1 - array2)
    print(f"像素差异最大值: {diff.max()}, 像素差异最小值: {diff.min()}")
    print(f"像素差异均值: {diff.mean()}, 像素差异中位数: {np.median(diff)}")

    # 是否完全相同
    if np.all(diff == 0):
        print("两幅图像完全相同！")
    else:
        print("两幅图像存在差异！")


# 示例用法
image1_path = "/home/iris/PycharmProjects/MambaCD/changedetection/datasets/data/mua2/test/T2/0003.png"  # 第一组图像路径
image2_path = "/home/iris/PycharmProjects/MambaCD/changedetection/datasets/data/mua2/test/T2/test_1.png"  # 第二组图像路径

compare_images(image1_path, image2_path)
