# Multi-Stage Human Intent Classifier System
This folder contains all the necessary files for predicting human intent classifier given a video clip.
To infer, please change the value of `video_path` inside the `inference.py` and simply run the 
```
python inference.py
```
Future edit will be done to support command line arguments.

Visualization can also generated using the `visualization.py`
```
python visualization.py
```
which will create a video output where the video input is annotated with the detected objects, and the gazed object.
