# Multi-Stage Human Intent Classifier System

This folder containsall the necessary files for creating a train-test split for both gaze-object classification and human intent classification tasks.

Use `see_annotate.ipynb` to visualize the annotations of an .MP4 video with an NDJSON annotation file generated used LabelBox.

Use `split_gaze.ipynb` to generate an inverse-frequency based random probabilistic train-test sampling on the entire dataset.

Use `split_intent.ipynb` to generate a random probabilistic train-test sampling with a fixed train sample size for each intention. The split can also follow from the `split_gaze.ipynb`.
