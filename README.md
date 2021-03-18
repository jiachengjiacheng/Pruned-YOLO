# Pruned-YOLO
Using model pruning method to obtain compact models, namely Pruned-YOLOv5, based on YOLOv5.

## Notice:

1.This project is based on [ultralytics/yolov5](https://github.com/ultralytics/yolov5). Place install it first. Then, use the model configuration file ( *coco_yolov5l.yaml* ) and network module definition file ( *common.py* ) provided here to replace the original corresponding files.

2.Referring to [SlimYOLOv3](https://github.com/PengyiZhang/SlimYOLOv3), we also use the subgradient method for sparsity training ( *sparsity.py* ). Besides, sparsity training and fine tuning are combined to simplify the pruning pipeline. We introduce the soft mask strategy and sparse factor cosine decay in the training process.

3.Use the *train_sr.py* for sparsity train, pruning can be done directly without fine-tuning.

4.Please place *prune_channel_v5_weightingByKernel.py* and *prune_layer_v5_weightingByKernel.py* in the home directory ( */yolov5/* ). The former is used for channel pruning and the latter for layer pruning. The model pruning can be done by them.

5.These two steps are iteratively applied until the compression model reaches the budget target.


## Results
![image](https://github.com/jiachengjiacheng/Pruned-YOLO/blob/main/results/ablationResults_VisDrone2018_valset.png)
Fig. 1: The ablation results of Pruned-YOLOv3 (left) and Pruned-YOLOv5 (right) on the VisDrone dataset. R, S and C represent the reweighting of channel important factors by convolution
kernel parameters, soft mask strategy and sparsity coefficient cosine decay respectively.

<img src="https://github.com/jiachengjiacheng/Pruned-YOLO/blob/main/results/results_VisDrone2018_valset.png" width="450" height="300" alt="lll"/><br/>
Fig. 2: Comparison of the Pruned-YOLO and other stateof-the-art object detectors on the VisDrone dataset. Pruned-YOLO has better parameter/accuracy trade-off than the others.

<img src="https://github.com/jiachengjiacheng/Pruned-YOLO/blob/main/results/results_coco2017_valset.JPG" width="427" height="357" alt="222"/><br/>
Fig. 3: Detection performance on the COCO 2017 val set. These models are compared in terms of parameters, BFLOPs and accuracy. Pruned-YOLOv5 achieves excellent performance in
the balance of parameters, calculation and accuracy.
