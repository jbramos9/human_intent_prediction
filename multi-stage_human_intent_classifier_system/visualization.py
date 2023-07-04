import torch
import torchvision
import numpy as np
import cv2

OBJECT_NAMES = ['Head', 'Bag', 'Book', 'Bottle', 'Bowl', 'Broom', 'Chair', 'Cup', 'Fruits', 'Laptop',
                'Pillow', 'Racket', 'Rug', 'Sandwich', 'Umbrella', 'Utensils']

from torchvision.ops import box_convert


def dataset_vis(img1, target, out=False):
    image = torchvision.transforms.ToPILImage('RGB')(img1[:3])
    img = np.array(image)
    # print('check_bbox()', img.shape)
    print(target['boxes'])
    for bix in range(0, len(target['boxes'])):
        bboxes = denormalize_box(target['boxes'][bix], image.size)

        x1, y1, x2, y2 = [int(a) for a in bboxes]
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 64, 0), 3)
        cv2.putText(img,
                    OBJECT_NAMES[target['objects'][bix]],
                    (x1 + 5, y1 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if target['isgazed'][bix] == 1 else (255, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(img, target['video_file'], (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(img, str(target['frame_number']), (5, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)

    # if out:
    #     img = cv2.resize(img, (1280, 720))
    #     out.write(img)

    cv2.imshow('show', img)
    cv2.waitKey(0)

def pred_vis(img1, pred, target, writer=None, show=True):

    if isinstance(img1, torch.Tensor):
        img1 = torchvision.transforms.ToPILImage('RGB')(img1[:3])
    img = np.array(img1)

    if len(img.shape) != 3:
        return

    h, w, c = img.shape

    boxes = pred["pred_boxes"][0]
    objects = torch.argmax(pred["pred_object_logits"], axis=2)[0]
    isgazed = torch.argmax(pred["pred_isgazed_logits"], axis=2)[0]

    head = {}
    gazed_object = {}

    gazed = translate(pred, "cuda")

    for bix in range(0, len(boxes)):
        bboxes = denormalize_box(boxes[bix], (w, h))
        x1, y1, x2, y2 = [int(a) for a in bboxes]

        if objects[bix] == 0:
            head["x1"] = x1
            head["x2"] = x2
            head["y1"] = y1
            head["y2"] = y2
        
        if objects[bix] == 16:
            continue

        if objects[bix]>=len(OBJECT_NAMES):
            continue
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 64, 0), 3)
        cv2.putText(img,
                    OBJECT_NAMES[objects[bix]],
                    (x1 + 5, y1 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 1, cv2.LINE_AA)
                    # cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if isgazed[bix] == 1 else (255, 0, 0), 1, cv2.LINE_AA)
        if objects[bix] == gazed and gazed!=16: #if isgazed[bix] == 1:
            gazed_object["x1"] = x1
            gazed_object["x2"] = x2
            gazed_object["y1"] = y1
            gazed_object["y2"] = y2

    if gazed_object:
        p1 = ((head["x1"] + head["x2"]) // 2, (head["y1"] + head["y2"]) // 2)
        p2 = ((gazed_object["x1"] + gazed_object["x2"]) // 2, (gazed_object["y1"] + gazed_object["y2"]) // 2)
        cv2.line(img, p1, p2, (0, 255, 0), 3)

    if target:
        true_objects = target[0]["objects"]
        true_boxes = target[0]["boxes"]
        true_gazed = target[0]["isgazed"].tolist()
        if 1 in true_gazed:
            true_gazed_idx = true_gazed.index(1)
            true_gazed_object = true_objects[true_gazed_idx]
            true_gazed_box = true_boxes[true_gazed_idx]
            # true_gazed_box = denormalize_box(true_gazed_box, (w, h))
            x1, y1, x2, y2 = [int(a) for a in true_gazed_box]
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 64, 255), 3)

        # img = cv2.resize(img, (1280, 720))
        cv2.putText(img,
                    f'{target[0]["video_file"].split("/")[-1][:-4]} - {target[0]["intent"]}',
                    (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)

    if writer:
        writer.write(img)

    if show:
        cv2.imshow('show', img)
        cv2.waitKey(1)
    
    return

def denormalize_box(box, image_size):
    width, height = image_size
    cx, cy, w, h = box
    bbox = torch.tensor([cx*width, cy*height, w*width, h*height])
    bbox = box_convert(bbox, 'cxcywh', 'xyxy')
    return bbox

def translate(outputs, device="cpu") -> torch.Tensor:
    scores = outputs["pred_isgazed_logits"].softmax(-1)
    isgazed = outputs["pred_isgazed_logits"].argmax(-1)

    mask = torch.where(isgazed == 1, isgazed, torch.zeros_like(isgazed))
    isgazed_scores = scores[:, :, 1] * mask

    objects = outputs["pred_object_logits"].argmax(-1)
    gazed_object = []
    for gaze, obj in zip(isgazed_scores, objects):
        if sum(gaze) == 0:
            gazed_object.append(torch.tensor(16).to(device))
        else:
            gazed_object_idx = gaze.argmax()
            gazed_object.append(obj[gazed_object_idx])
    gazed_object = torch.stack(gazed_object)
    return gazed_object