import os

import cv2
from PIL import Image
import imagehash

from ultralytics import YOLO


if __name__ == '__main__':
    model_ori = YOLO("runs/detect/train31/weights/best.pt")
    model_opt = YOLO("runs/detect/train30/weights/best.pt")

    # img_path = "./data/DETRAC-train-data/Insight-MVT_Annotation_Train/MVI_20012/img00032.jpg"
    # results = model.predict(source=img_path)
    # print(results[0])
    # res_plot = results[0].plot()
    # cv2.imshow("test", res_plot)
    # cv2.waitKey(0)

    # origin_img = results[0].orig_img
    # boxes = results[0].boxes
    # get_result(origin_img, boxes)

    test_case = 'MVI_40793'
    pic_path = f"./data/DETRAC-test-data/Insight-MVT_Annotation_Test/{test_case}/"

    for img in os.listdir(pic_path):
        img_path = os.path.join(pic_path, img)
        results_ori = model_ori.predict(source=img_path)
        results_opt = model_opt.predict(source=img_path)
        results_ori_plot = results_ori[0].plot()
        results_opt_plot = results_opt[0].plot()
        # compare content
        if results_ori_plot.shape != results_opt_plot.shape:
            print(f"shape not equal: {img_path}")
            continue
        results_img_ori = Image.fromarray(results_ori_plot)
        results_img_opt = Image.fromarray(results_opt_plot)
        dhash_ori = imagehash.dhash(results_img_ori)
        dhash_opt = imagehash.dhash(results_img_opt)
        if dhash_ori - dhash_opt > 5:
            print(f"content not equal: {img_path}")
            # save img
            cv2.imwrite(f"./videos/{test_case}_{img[:img.rfind('.')]}_ori.jpg", results_ori_plot)
            cv2.imwrite(f"./videos/{test_case}_{img[:img.rfind('.')]}_opt.jpg", results_opt_plot)
