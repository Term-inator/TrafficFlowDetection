import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt
from video import get_result

# Load a model
model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)


if __name__ == '__main__':
    # Use the model
    model.train(data="train.yaml", epochs=50)  # train the model
    metrics = model.val()  # evaluate model performance on the validation set
    # print(metrics)
    # path = model.export(format="onnx")  # export the model to ONNX format
    # print(path)
    # results = model("./data/images/test/MVI_39031_img00501.jpg")  # predict on an image
    # res_plot = results[0].plot()
    # cv2.imshow("test", res_plot)
    # cv2.waitKey(0)
