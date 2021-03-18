# Pruned-YOLO
Using model pruning method to obtain compact models Pruned-YOLOv5 based on YOLOv5.

## NOTICE:

1.This project is based on *ultralytics/yolov5(https://github.com/ultralytics/yolov5)*. Install it first. Then, use the model configuration file (*coco_yolov5l.yaml*) and network module definition file (*common.py*) provided here to replace the original corresponding files.

2.Please place *prune_channel_v5_weightingByKernel.py* and *prune_layer_v5_weightingByKernel.py* in the home directory (*/yolov5/*). The former is used for channel pruning and the latter for layer pruning. The model pruning can be done by them.

3.Referring to *SlimYOLOv3(https://github.com/PengyiZhang/SlimYOLOv3)*, we also use the subgradient method for sparsity training (*sparsity.py*). Besides, sparsity training and fine tuning are combined to simplify the pruning pipeline. We introduce the soft mask strategy and sparse factor cosine decay in the training process (*train.py*).
