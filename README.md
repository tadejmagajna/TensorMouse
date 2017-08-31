# TensorMouse
Control your mouse cursor by moving objects in front of webcam using Tensorflow Object Detection API

TensorMouse allows you to control your cursor by moving a random objects (like cups, apples or bananas) in front of webcam to move your cursor as a replacement for mouse or touchpad.

Tensorflow Object Detection API trained on the COCO dataset supports up to 90 different types of objects that can be used to move the cursor.
 
Project includes ssd_mobilenet_v1_coco_11_06_2017 Tensorflow frozen graph trained on the mobilenet deep neural network by default, but supports any other graph Tensorflow Object Detection API graph by supplying the optional argument `--graphpath`

TensorMouse also supports clicks currently triggered by clicking CTRL key and mouse drags triggered by ALT key.

## Getting Started
1. `conda env create -f environment.yml`
2. Activate environment:
    * Windows: `activate tensormouse`
    * Unix: `source activate tensormouse`
2. `python tensormouse.py` 
    Optional arguments (default value):
    * Device index of the camera `--source=0`
    * Object name - name of object to track (see graphs/labels.json for available options) `--object=cup`
    * Path to frozen tensofrlow graph `--graphpath=ssd_mobilenet_v1_coco_11_06_2017`
2. Wait for `✓ TensorMouse started successfully!` message to use the application. Use CTRL for clicks and ALT for cursor draging.
Exit by clicking CAPS_LOCK

## Requirements
- [Anaconda / Python 3.5](https://www.continuum.io/download)
- [TensorFlow 1.3](https://www.tensorflow.org/)
- [OpenCV 3.1](http://opencv.org/)

## Technologies Used
- [Tensorflow Object Detection API](https://github.com/tensorflow/models/tree/master/object_detection)
- [COCO dataset](http://mscoco.org/dataset/)
- datitran/object_detector_app

## Notes
- Make sure you have good lighting when using TensorMouse
- Object deteciton on screen edges is poor. TODO: scale cursor movement so that near edge object movements will move cursor to edge

## Copyright

See [LICENSE](LICENSE) for details.
Copyright (c) 2017 [Tadej Magajna](http://www.tadejmagajna.com/).
