import os
from PIL import Image

def convert_to_rgb(input_path, output_path):
    """
    将 PNG 图像转换为 RGB 格式，并保存到指定路径。

    参数:
    - input_path (str): 输入图像路径。
    - output_path (str): 转换后保存的图像路径。
    """
    # 检查输入路径是否存在
    if not os.path.exists(input_path):
        print(f"输入路径不存在: {input_path}")
        return

    # 打开图像
    image = Image.open(input_path)

    # 检查是否为 4 通道 (RGBA)
    if image.mode == 'RGBA':
        print("检测到 4 通道 (RGBA) 图像，正在转换为 RGB...")
        image = image.convert('RGB')  # 转换为 RGB
    elif image.mode != 'RGB':
        print(f"检测到非 RGB 模式图像 ({image.mode})，正在转换为 RGB...")
        image = image.convert('RGB')
    else:
        print("图像已是 RGB 模式，无需转换。")

    # 保存转换后的图像
    os.makedirs(os.path.dirname(output_path), exist_ok=True)  # 确保输出目录存在
    image.save(output_path)
    print(f"转换后的 RGB 图像已保存到: {output_path}")


# 示例用法
if __name__ == "__main__":
    input_image_path = "/home/iris/PycharmProjects/MambaCD/changedetection/datasets/data/mua2/test/T2/0001.png"  # 输入图像路径
    output_image_path = "/home/iris/PycharmProjects/MambaCD/changedetection/datasets/data/mua2/test/T2/0005.png"  # 输出图像路径
    convert_to_rgb(input_image_path, output_image_path)
