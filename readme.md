## Pytorch Keypoint Detection

#### 介绍

2019.05月pytorch发布了torchvision0.3, 里面实现了Mask_RCNN, Keypoint_RCNN和DeepLabV3，可以直接用于语义分割，目标检测，实例分割和人体关键点检测4个任务。

在github上torch/vision/reference里面有classification, detection和segmentation三个文件夹，分别对应不同任务。直接运行detection的代码是MaskRCNN的实现，用于目标检测和实例分割任务。官网也有对应的[教程](https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html)可以轻松通过COCO2017数据集进行训练和测试。

但是如果要实现人体关键点检测的话需要在detection的文件修改一些参数，这里我将修改后的文件和可视化程序附上。

![img1](https://github.com/scnuhealthy/Pytorch-Keypoint-Detection/blob/master/result/10.jpg)

![img2](https://github.com/scnuhealthy/Pytorch-Keypoint-Detection/blob/master/result/14.jpg)

![img3](https://github.com/scnuhealthy/Pytorch-Keypoint-Detection/blob/master/result/5.jpg)

![img4](https://github.com/scnuhealthy/Pytorch-Keypoint-Detection/blob/master/result/8.jpg)

![img5](https://github.com/scnuhealthy/Pytorch-Keypoint-Detection/blob/master/result/9.jpg)

#### Dataset

下载COCO2017数据集，下载train2017,val2017和annotations三个文件后解压，最终文件目录结构如下

COCO2017/

​	train2017/

​	val2017/

​	annotations/

#### Train

对`train.py`进行如下操作

1 修改函数`get_dataset`中的`paths`

2 修改文件中列出了的各种参数

3 假设路径都修改完毕，使用预训练模型进行训练: `python train.py --pretrained`

#### Predict and Visualize

对COCO val集的图片进行预测并可视化。

对predict_visualize.py进行如下操作

1 修改代码中的路径

2 修改参数`detect_threshold`和`keypoint_score_threshold`, 分别过滤得分低的个体和得分低的关键点

3 在根目录下建立文件夹result, 可视化后的图片存在在此文件夹下

4 运行`python predict_visualize.py`

#### Evaluate

执行代码`python train.py --test-only`

程序在COCO2017 val集上的结果如下，同[官网介绍](https://pytorch.org/blog/torchvision03/)一致

```
Averaged stats: model_time: 0.1371 (0.1627)  evaluator_time: 0.0043 (0.0105)
Accumulating evaluation results...
DONE (t=1.31s).
Accumulating evaluation results...
DONE (t=0.41s).
IoU metric: bbox
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.502
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.796
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.545
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.341
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.591
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.648
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.176
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.519
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.603
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.460
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.669
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.738
IoU metric: keypoints
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.599
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.834
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.650
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.553
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.675
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.672
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.889
 Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.721
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.623
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.741
```





