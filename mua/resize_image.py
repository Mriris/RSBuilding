import os
from PIL import Image

def resize_image(input_path, output_path, target_size=(1024, 1024)):
    """
    调整图像大小，并保存到指定路径。

    参数:
    - input_path (str): 输入图像路径。
    - output_path (str): 调整后保存的图像路径。
    - target_size (tuple): 调整图像的目标大小 (默认: 1024x1024)。
    """
    # 检查输入路径是否存在
    if not os.path.exists(input_path):
        print(f"输入路径不存在: {input_path}")
        return

    # 打开图像
    image = Image.open(input_path)

    # 调整图像大小
    print(f"正在调整图像大小到: {target_size}")
    image = image.resize(target_size, Image.Resampling.LANCZOS)  # 使用 LANCZOS 作为高质量的重采样方法

    # 保存调整后的图像
    os.makedirs(os.path.dirname(output_path), exist_ok=True)  # 确保输出目录存在
    image.save(output_path)
    print(f"调整后的图像已保存到: {output_path}")


# 示例用法
if __name__ == "__main__":
    input_image_path = "/home/iris/Datasets/mua4/origin/Y10.png"  # 输入图像路径
    output_image_path = "/home/iris/Datasets/mua4/origin/Y12.png"  # 输出图像路径
    resize_image(input_image_path, output_image_path, target_size=(1024, 1024))
