# Multi-Stage Hybrid-CNN Transformer Model for Human Intent-Prediction
This repo contains all the models (including the variant) developed in the project [ Multi-Stage Hybrid-CNN Transformer Model for Human Intent-Prediction](). As an overview, the Multi-Stage Hybrid-CNN Transformer Classifier System is composed of two key components: the Gazed Object Detector and the Intent Classifier.
Three version are present for the Gazed Object Detector:
- The `modified_detr` version is the default single-stream model
- the `modified_detr_no_depth` version is the version that disregards the depth information
- and the `modified_detr_head` version the multi-stream model that requires another head detector model for the head input

The Intent Classifier is in the `intent_classifier` folder.
Additionaly, the overall system for inference is in the `multi-stage_human_intent_classifier_system` folder. To learn more each about each model, each folder contains its own readme.md for guidance.
