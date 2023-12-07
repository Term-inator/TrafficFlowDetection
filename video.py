# 把某个文件夹下的图片合成视频
import os
import shutil

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import matplotlib.pyplot as plt


cls_dict = {0: 'others', 1: 'car', 2: 'van', 3: 'bus'}
def get_result(results):
    orig_img = results[0].orig_img
    boxes = results[0].boxes
    cls = boxes.cls.to('cpu').numpy()
    for i, box in enumerate(boxes.xyxy):
        box = box.to('cpu').numpy().astype(np.int32)
        cv2.rectangle(orig_img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), thickness=1)
        cv2.putText(orig_img, str(cls_dict[cls[i]]), (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    return orig_img


def pic2video(pic_path, video_path, fps, predict_func):
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    images = os.listdir(pic_path)
    im = Image.open(pic_path + images[0])
    vw = cv2.VideoWriter(video_path, fourcc, fps, im.size)
    for i in range(len(images)):
        im = Image.open(os.path.join(pic_path, images[i]))
        im_cv = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)
        results = predict_func(im_cv)
        im_cv = get_result(results)
        vw.write(im_cv)
    vw.release()
    print('Finish!')


if __name__ == '__main__':
    model = YOLO("runs/detect/train31/weights/best.pt")
    # model.eval()

    # img_path = "./data/DETRAC-train-data/Insight-MVT_Annotation_Train/MVI_20012/img00032.jpg"
    # results = model.predict(source=img_path)
    # print(results[0])
    # res_plot = results[0].plot()
    # cv2.imshow("test", res_plot)
    # cv2.waitKey(0)

    # origin_img = results[0].orig_img
    # boxes = results[0].boxes
    # get_result(origin_img, boxes)

    test_case = 'MVI_40743'
    pic_path = f"./data/DETRAC-test-data/Insight-MVT_Annotation_Test/{test_case}/"
    if not os.path.exists('videos'):
        os.makedirs('videos')
    video_path = f'./videos/{test_case}.avi'
    fps = 25
    pic2video(pic_path, video_path, fps, lambda x: model.predict(source=x))
