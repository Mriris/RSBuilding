import os
from multiprocessing import Pool
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-img-dir", default="data_dir/whubuilding/train/images")
    parser.add_argument("--input-mask-dir", default="data_dir/whubuilding/train/gt")
    parser.add_argument("--output-txt-dir", default="data_dir/whubuilding/train.txt")
    return parser.parse_args()


def process_file(filename):
    # 构建对应的 B 和标签文件名
    pic_filename = filename
    label_filename = filename.split('.')[0] + '.tif'
    if not (label_filename in label_filenames):
        return None
    a_path = os.path.join(img_folder, pic_filename)
    label_path = os.path.join(label_folder, label_filename)
    return f'{a_path}\t**\t**\t{label_path}\t**\n'


if __name__ == '__main__':
    args = parse_args()
    img_folder = args.input_img_dir
    label_folder = args.input_mask_dir
    txt_dir = args.output_txt_dir

    print("接受的参数为：", args)

    # 获取文件列表并按字母排序
    a_filenames = sorted(os.listdir(img_folder))
    label_filenames = sorted(os.listdir(label_folder))

    with Pool(16) as p:
        results = p.map(process_file, a_filenames)

    # 过滤 None 并将结果写入 txt 文件
    with open(txt_dir, 'w') as f:
        for result in results:
            if result is not None:
                f.write(result)
