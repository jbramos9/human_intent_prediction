# Intent Classifier
This folder contains all the necessary files to train, test and evaluate the intent classifier model.

## Train
Before training, please configure first the `config.yaml` file if you want to use the config file. Our `train.py` supports both using the config file or by passing arguments in the command line. We recommend to use the config.yaml for easy configuration.

To train using the `config.yaml` file,
```
python train.py --use_config --run_name any_run_name_you_want
```

## Test
To test, please change the ckpt path appropriately in the following command,
```
python test.py --ckpt_dir "checkpoints/small_model.ckpt"
```

## Evaluation
To evaluate,
```
python eval.py --ckpt_dir "checkpoints/small_model.ckpt" --device "cuda" --gaze_outputs "gaze_outputs/with_head_pretrained.csv"
```
where the `gaze_output.csv` contains the output of the gaze object detector.
