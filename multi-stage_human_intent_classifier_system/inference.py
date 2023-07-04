import yaml

from models.mhics import MHICS


if __name__=="__main__":

    with open("gaze_config.yaml", "r") as f:
        gaze_config = yaml.safe_load(f)
    gaze_ckpt = r"trained_weights\with_head_pretrained.ckpt"

    with open("intent_config.yaml", "r") as f:
        intent_config = yaml.safe_load(f)
    intent_ckpt = r"trained_weights\3-encoders.ckpt"

    IntentClassifier = MHICS(gaze_config, gaze_ckpt, intent_config, intent_ckpt, device="cuda")

    video_path = r"VIDEOS/P16/V9/P16_V9.mp4"
    intent = IntentClassifier(video_path)
    print(intent)