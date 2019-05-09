# pedestrian-counting
pedestrian counting solution for my Master Thesis - Usage of convolutional neural networks for pedestrian flow analysis.

Tensorflow Object detection models are used in this script to perform pedestrian detection in each frame.

Detection models need to be downloaded from Tensorflow object detection model zoo found here: github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md

Most simple way is to use MobileNetSSD model for inference, because it is lightweight and runs pretty fast on CPU.
