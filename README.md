# docker-makefile
This repo consists of several tutorials, which use Docker+make for some CS-courses.

### docker_tutorials
This package is forked from the repo of [RAIL-software-infrastructure-demos](https://github.com/RAIL-group/RAIL-software-infrastructure-demos). It is used in running simple Unity with GPU, Python, Cpp program, Parallel Program, PyBind.


### ubuntu18
Ubuntu 18.04 is installed with cude 12.0.0.
This package initialized for cs695-005 course's project, and ran several networks to process point cloud and traffic sign regconition.

Due to the large size of the models, the models are not uploaded to the github. Please download the models from the following links and put them in the *ubuntu18/script/regconition2/traffic_models* folder for Alexnet model, and *ubuntu18/scripts/pcl/complex_yolov4* folder for complex-yolov4 model.
- [faster_rcnn_resnet101_coco_11_06_2017 model](http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet101_coco_11_06_2017.tar.gz)
- [complex-yolov4 model](https://drive.google.com/drive/folders/1RHD9PBvk-9SjbKwoi_Q1kl9-UGFo2Pth)

### ubuntu20
Ubuntu 20.04 is installed with cude 11.3.1 and openGL.
This package initialized for cs695-005 course's project, too; and ran several networks to process traffic sign detection and traffic light detection-regconition. Due to the large size of the models, the models are not uploaded to the github. Please download the models from the following links and put them in the *models* folder.
- [Alexnet model](https://github.com/surajmurthy/TSR_PyTorch/blob/main/Model/pytorch_classification_alexnetTS.pth)
- [faster_rcnn_inception_resnet_v2_atrous model](https://drive.google.com/open?id=12vLvA9wyJ9lRuDl9H9Tls0z5jsX0I0Da) 

Many other tensorflow models are available at the following [link](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md )

### Acknowledgement
We have learned and used the following repos to build this repo. Thank you for their great work.
- https://github.com/nileshchopda/Traffic-Light-Detection-And-Color-Recognition
- https://github.com/aarcosg/traffic-sign-detection
- https://github.com/RAIL-group/RAIL-software-infrastructure-demos
- https://github.com/surajmurthy/TSR_PyTorch

