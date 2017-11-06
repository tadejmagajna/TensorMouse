# TensorMouse


Control your mouse cursor by moving objects in front of webcam using Tensorflow Object Detection API.

![alt text](https://github.com/tadejmagajna/TensorMouse/blob/master/docs/preview.png?raw=true)

TensorMouse allows you to control your cursor by moving random objects (like cups, apples or bananas) in front of webcam to move your cursor as a replacement for mouse or touchpad.

Tensorflow Object Detection API trained on the COCO dataset supports up to 90 different types of objects that can be used to move the cursor.
 
Project includes ssd_mobilenet_v1_coco_11_06_2017 Tensorflow frozen graph trained on the mobilenet deep neural network by default, but supports any other graph Tensorflow Object Detection API graph by supplying the optional argument `--graphpath`

TensorMouse also supports clicks currently triggered by clicking CTRL key and mouse drags triggered by ALT key.



## Getting Started
### Linux/MacOS
1. Make sure you have anaconda installed
2. Open a terminal and navigate to project root
3. `conda env create -f environment.yml`
4. `source activate tensormouse`
### Windows
1. Make sure you have anaconda installed
2. Open a terminal and navigate to project root
3. `conda env create -f environment-win.yml`
4. `activate tensormouse`

## Usage
```
python tensormouse.py
```
Optional arguments (default value):
 * `--source=0` Device index of the camera
 * `--object=cup` Object name - name of object to track (see graphs/labels.json for available options) 
 * `--graphpath=ssd_mobilenet_v1_coco_11_06_2017` Path to frozen tensofrlow graph 


Wait for `✓ TensorMouse started successfully!` message to use the application.

Use CTRL for clicks and ALT for cursor draging.
Exit by clicking CAPS_LOCK

## Requirements
- [Anaconda / Python 3.5](https://www.continuum.io/download)
- [TensorFlow 1.3](https://www.tensorflow.org/)
- [OpenCV 3.1](http://opencv.org/)

## Technologies Used
- [Tensorflow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection)
- [COCO dataset](http://mscoco.org/dataset/)
- datitran/object_detector_app

## Notes
- Make sure you have good lighting when using TensorMouse
- Frame rate will be very low on low performing machines
- Object deteciton on screen edges is poor. (todo: scale cursor movement so that near edge object movements will move cursor to edge)

## Copyright

See [LICENSE](LICENSE) for details.
Copyright (c) 2017 [Tadej Magajna](http://www.tadejmagajna.com/).
