import os

from ultralytics import YOLO


def model_factory(conv_type=None, attention_type=None, pretrained=True):
    data_path = os.path.join('train_yaml', 'train.yaml')
    if conv_type is None and attention_type is None:
        model_path = 'yolov8n.yaml'
        cfg_path = os.path.join('train_yaml', 'train_cfg.yaml')
    else:
        type_lst = []
        if conv_type is not None:
            type_lst.append(conv_type)
        if attention_type is not None:
            type_lst.append(attention_type)

        filename = '_'.join(type_lst)
        model_path = os.path.join('models', filename + '.yaml')
        cfg_path = os.path.join('train_yaml', filename + '_cfg.yaml')

    model = YOLO(model_path)

    if pretrained:
        model.load("yolov8n.pt")

    return model, cfg_path, data_path


if __name__ == '__main__':
    model, cfg_path, data_path = model_factory(conv_type='transformer', attention_type='coord', pretrained=False)
    print(cfg_path, data_path)
    # Use the model
    model.train(cfg=cfg_path, data=data_path, epochs=100)  # the model
    metrics = model.val()  # evaluate model performance on the validation set
    # print(metrics)
    # path = model.export(format="onnx")  # export the model to ONNX format
    # print(path)
    # results = model("./data/images/test/MVI_39031_img00501.jpg")  # predict on an image
    # res_plot = results[0].plot()
    # cv2.imshow("test", res_plot)
    # cv2.waitKey(0)
