# Multi-Stage Hybrid-CNN Transformer Model for Human Intent-Prediction
This repo contains all the models (including the variant) developed in the project [Multi-Stage Hybrid-CNN Transformer Model for Human Intent-Prediction](https://drive.google.com/file/d/1tVHbOet_5-99KdBAT0jSuujO2GBRi-nq/view?usp=sharing). As an overview, the Multi-Stage Hybrid-CNN Transformer Classifier System is composed of two key components: the Gazed Object Detector and the Intent Classifier.
- The Gazed Object Detector is in the `gazed_object_detectors` which contains the three different variations of the model.
- The Intent Classifier is in the `intent_classifier` folder.
- The Overall System **for inference** is in the `multi-stage_human_intent_classifier_system` folder.
Each folder have their own readme.md for guidance.

# Dataset
The dataset that was used in this project can be accessed [here](https://drive.google.com/drive/folders/1L4mau-UvI51qa2JSlboCKberItyMLiIl?usp=sharing). The generators and statistics for the train-test split are in the `split` folder.

# Recommendations
1. Add more video samples such that the gaze distribution is balanced ("None" or not looking at objects is currently overrepresented)
2. Develop the weights of the Gaze Object Detector from scratch to tailor the model for object-gaze classification
3. Consider the probability of gaze for all objects in a given frame, instead of the most probable gaze, as input to the human intent classifier
4. Explore other human pose estimation techniques (as inspired by the increase in performance from the additional head information used)
