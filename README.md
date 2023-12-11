# TrafficFlowDetection

Final Project for COMP 576

- /models/transformer_coord.yaml: model configuration file of TRCA-YOLOv8
- /train_yaml/transformer_coord_cfg.yaml: training configuration file of TRCA-YOLOv8
- /ultralytics/nn/modules/block.py: contains implementation of C2fTR block
- /ultralytics/nn/modules/coord_attention.py: contains implementation of [Coordinate Attention](https://github.com/houqb/CoordAttention)
- compare.py: compare the performance of TRCA-YOLOv8 and YOLOv8
- process.py: transform the dataset to the format of YOLOv8
- track.py: track the vehicles in the video with BoT-SORT
- train.py: train the model
- video.py: apply the model to the dataset and generate the video
