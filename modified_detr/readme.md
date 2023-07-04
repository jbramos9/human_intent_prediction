# Gazed Object Detector
This folder contains all the files needed for training and testing the model.
-----
**_NOTE:_** Since very large files are not yet uploaded in this repo. Please download these folders for the meantime in this [link](). These folders should be inside each folder of the variation.
- `data` which should contain the pretrained_weights, the gaze_dataset json files.
- `checkpoints` which should contain all the trained weights in the model
-----
Since, the training and testing framework are very similar to all the different versions, same procedures were done to train each variation. Three versions are present for the Gazed Object Detector:
- The `modified_detr` version is the default single-stream model
- the `modified_detr_no_depth` version is the version that disregards the depth information
- and the `modified_detr_head` version the multi-stream model that requires another head detector model for the head input
## Train
Before proceeding to training, please make sure that `config.yaml` is properly configured. To train simply run
```trainer.py```
## Testing
Same with training, please make sure that `config.yaml` is properly configured. In addiition, please also change the checkpoint file in the following line
```model = LITGaMer.load_from_checkpoint("checkpoints/with_head_pretrained.ckpt")```
To test simply run
```test.py```
