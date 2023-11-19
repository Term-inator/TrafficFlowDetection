import os

from ultralytics import YOLO


def model_factory(attention_type=None, conv_type=None, pretrained=True):
    model = YOLO("yolov8n.yaml")
    cfg_path = os.path.join('train_yaml', 'default.yaml')
    data_path = os.path.join('train_yaml', 'train.yaml')
    if attention_type == 'coord':
        model = YOLO("models/coord.yaml")
        cfg_path = os.path.join('train_yaml', 'coord_cfg.yaml')

    if pretrained:
        model.load("yolov8n.pt")

    return model, cfg_path, data_path


if __name__ == '__main__':
    model, cfg_path, data_path = model_factory(pretrained=True)
    print(cfg_path, data_path)
    # Use the model
    model.train(cfg=cfg_path, data=data_path, epochs=50)  # the model
    metrics = model.val()  # evaluate model performance on the validation set
    # print(metrics)
    # path = model.export(format="onnx")  # export the model to ONNX format
    # print(path)
    # results = model("./data/images/test/MVI_39031_img00501.jpg")  # predict on an image
    # res_plot = results[0].plot()
    # cv2.imshow("test", res_plot)
    # cv2.waitKey(0)
