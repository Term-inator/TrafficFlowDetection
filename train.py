from ultralytics import YOLO
import matplotlib.pyplot as plt

# Load a model
model = YOLO("yolov8m.pt")  # load a pretrained model (recommended for training)


if __name__ == '__main__':
    # Use the model
    model.train(data="train.yaml", epochs=1)  # train the model
    metrics = model.val()  # evaluate model performance on the validation set
    print(metrics)
    path = model.export(format="onnx")  # export the model to ONNX format
    print(path)
    results = model("./data/images/test/MVI_39031_img00501.jpg")  # predict on an image
    print(results)
