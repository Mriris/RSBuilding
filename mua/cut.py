from PIL import Image
import os

def crop_center_to_square(input_path, output_path, size=(512, 512)):
    """
    从图像的中心裁剪指定大小的正方形部分，并保存到输出路径。

    参数:
    - input_path (str): 输入图像路径。
    - output_path (str): 输出图像保存路径。
    - size (tuple): 裁剪的目标大小 (宽, 高)，默认是 (1024, 1024)。
    """
    # 检查输入路径是否存在
    if not os.path.exists(input_path):
        print(f"输入图像路径不存在: {input_path}")
        return

    # 打开图像
    image = Image.open(input_path)

    # 获取图像尺寸
    width, height = image.size
    print(f"原图尺寸: {width}x{height}")

    # 确保目标尺寸不超过原图尺寸
    target_width, target_height = size
    if target_width > width or target_height > height:
        print("目标尺寸超过原图尺寸，无法裁剪！")
        return

    # 计算中心裁剪区域
    left = (width - target_width) // 2
    top = (height - target_height) // 2
    right = left + target_width
    bottom = top + target_height
    print(f"裁剪区域: 左={left}, 上={top}, 右={right}, 下={bottom}")

    # 裁剪图像
    cropped_image = image.crop((left, top, right, bottom))

    # 保存裁剪后的图像
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cropped_image.save(output_path)
    print(f"裁剪后的图像已保存到: {output_path}")


# 示例用法
if __name__ == "__main__":
    input_image_path = "/home/iris/Datasets/mua2/O12.tif"  # 输入图像路径
    output_image_path = "/home/iris/Datasets/mua2/test/O20.tif"  # 输出图像路径
    crop_center_to_square(input_image_path, output_image_path)
