import cv2
import numpy as np
import os
from multiprocessing import Pool
from tqdm import tqdm
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="data_dir/Levir-CD/train/A")
    parser.add_argument("--output-dir", default="data_dir/Levir-CD/train/A_512_nooverlap")
    return parser.parse_args()


def slide_crop(filename):
    real_path = os.path.join(folder_path, filename)
    name, ext = os.path.splitext(filename)  # 分离文件名和扩展名
    img = cv2.imread(real_path)
    crop_width, crop_height = 512, 512
    step_x, step_y = 512, 512
    height, width, _ = img.shape
    crops = []
    for i in range(0, width - crop_width + 1, step_x):
        for j in range(0, height - crop_height + 1, step_y):
            crop = img[j:j + crop_height, i:i + crop_width]
            crops.append(crop)
    for i, crop in enumerate(crops):
        # 使用输入文件的扩展名生成输出文件
        cv2.imwrite(f'{out_path}/{name}_crop_{i}{ext}', crop)


if __name__ == '__main__':
    args = parse_args()
    folder_path = args.input_dir
    out_path = args.output_dir
    os.makedirs(out_path, exist_ok=True)

    filenames = [filename for filename in os.listdir(folder_path) if
                 filename.endswith('.png') or filename.endswith('.jpg') or filename.endswith('.tif') or filename.endswith('.tiff')]

    with Pool(16) as p:
        list(tqdm(p.imap(slide_crop, filenames), total=len(filenames)))
