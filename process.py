import argparse
import os
import shutil
from xml.dom.minidom import parse
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor


def parse_frame(frame, scene_name, labels_root):
    img_w = 960
    img_h = 540
    seq_gt = []

    frame_num = frame.getAttribute("num")
    frame_num = frame_num.zfill(5)
    label_name = scene_name + '_img' + frame_num

    img_root = './data/DETRAC/images/train/' if 'train' in labels_root else './data/DETRAC/images/test/'
    sample_img = os.listdir(img_root)
    if (label_name + '.jpg') in sample_img:
        label_path = os.path.join(labels_root, label_name + '.txt')

        targets = frame.getElementsByTagName('target_list')[0].getElementsByTagName('target')
        for target in targets:
            box = target.getElementsByTagName('box')[0]
            left = float(box.getAttribute('left'))
            top = float(box.getAttribute('top'))
            width = float(box.getAttribute('width'))
            height = float(box.getAttribute('height'))
            type = target.getElementsByTagName('attribute')[0].getAttribute('vehicle_type')
            if type == "car":
                type = 1
            elif type == "van":
                type = 2
            elif type == "bus":
                type = 3
            else:
                type = 0

            # Transform the bbox co-ordinates as per the format
            x = left + width / 2
            y = top + height / 2

            # Normalise the co-ordinates by the dimensions of the image
            x /= img_w
            y /= img_h
            width = width / img_w
            height = height / img_h

            # seq_gt.append(str(type) + ' ' + str(round(x, 6)) + ' ' + str(round(y, 6)) + ' ' + str(
            #     round(width, 6)) + ' ' + str(round(height, 6)))
            seq_gt.append(f'{type} {x:.6f} {y:.6f} {width:.6f} {height:.6f}')

        with open(label_path, 'w') as f:
            for i in seq_gt:
                f.write(i + '\n')


def create_labels(root_node, labels_root, thread_num=8):
    frames = root_node.getElementsByTagName('frame')

    if thread_num == 1:
        for frame in frames:
            scene_name = frame.parentNode.getAttribute('name')
            parse_frame(frame, scene_name, labels_root)
    else:
        with ThreadPoolExecutor(max_workers=thread_num) as executor:
            futures = []
            for frame in frames:
                scene_name = frame.parentNode.getAttribute('name')
                futures.append(executor.submit(parse_frame, frame, scene_name, labels_root))

            for future in futures:
                future.result()


def yolo_img(params, mode='train'):
    img_dir = f'./data/DETRAC/images/{mode}/'
    if os.path.exists(img_dir):
        shutil.rmtree(img_dir)
    os.makedirs(img_dir)
    dataset_dir = params.img_train if mode == 'train' else params.img_test

    with tqdm(total=len(os.listdir(dataset_dir))) as pbar:
        pbar.set_description(f'YOLO {mode} set image generating')
        for video_name in os.listdir(dataset_dir):
            if 'MVI' in video_name:
                stride = int(1.0 / params.sample)
                for i, img_name in enumerate(os.listdir(os.path.join(dataset_dir, video_name))):
                    if ((i % stride) == 0) & ('.jpg' in img_name):
                        ori_path = os.path.join(dataset_dir, video_name, img_name)
                        new_path = os.path.join(img_dir, video_name + '_' + img_name)
                        shutil.copyfile(ori_path, new_path)
                pbar.update(1)


def yolo_label(params, mode='train', thread_num=8):
    label_dir = f'./data/DETRAC/labels/{mode}/'
    if os.path.exists(label_dir):
        shutil.rmtree(label_dir)
    os.makedirs(label_dir)
    dataset_dir = params.lbl_train if mode == 'train' else params.lbl_test

    with tqdm(total=len(os.listdir(dataset_dir))) as pbar:
        pbar.set_description(f'YOLO {mode} set label generating')
        for video_xml in os.listdir(dataset_dir):
            if '.xml' in video_xml:
                create_labels(parse(os.path.join(dataset_dir, video_xml)).documentElement, label_dir, thread_num)
                pbar.update(1)


def gen_yolo_dataset(params):
    yolo_img(params, mode='train')
    yolo_img(params, mode='test')

    yolo_label(params, mode='train', thread_num=1)
    yolo_label(params, mode='test', thread_num=1)


def parse_params():
    parser = argparse.ArgumentParser(prog='process.py')
    parser.add_argument('--img_train', type=str, default='./data/DETRAC-train-data/Insight-MVT_Annotation_Train/', help='train images path')
    parser.add_argument('--img_test', type=str, default='./data/DETRAC-test-data/Insight-MVT_Annotation_Test/', help='test images path')
    parser.add_argument('--lbl_train', type=str, default='./data/DETRAC-Train-Annotations-XML/', help='train labels path')
    parser.add_argument('--lbl_test', type=str, default='./data/DETRAC-Test-Annotations-XML/', help='test labels path')
    parser.add_argument('--sample', type=float, default=1.0, help='equidistant sampling ratio')
    params = parser.parse_args()
    return params


if __name__ == "__main__":
    params = parse_params()
    params.sample = 0.025
    gen_yolo_dataset(params)





