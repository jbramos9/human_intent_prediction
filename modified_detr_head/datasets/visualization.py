import torch
import torchvision
import numpy as np
import cv2

from datasets.gom import OBJECT_NAMES

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

def pred_vis(img1, target, out=False):
    if isinstance(img1, torch.Tensor):
        img1 = torchvision.transforms.ToPILImage('RGB')(img1[:3])
    img = np.array(img1)
    h, w, c = img.shape

    boxes = target["pred_boxes"][0]
    objects = torch.argmax(target["pred_object_logits"], axis=2)[0]
    isgazed = torch.argmax(target["pred_isgazed_logits"], axis=2)[0]
    # print("boxes:", boxes)
    # print("objects:", objects)
    # print("isgazed:", isgazed)
    for bix in range(0, len(boxes)):
        bboxes = denormalize_box(boxes[bix], (w, h))

        x1, y1, x2, y2 = [int(a) for a in bboxes]
        if objects[bix]>=len(OBJECT_NAMES):
            continue
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 64, 0), 3)
        cv2.putText(img,
                    OBJECT_NAMES[objects[bix]],
                    (x1 + 5, y1 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if isgazed[bix] == 1 else (255, 0, 0), 1, cv2.LINE_AA)

    # if out:
    #     img = cv2.resize(img, (1280, 720))
    #     out.write(img)
    img = cv2.resize(img, (1280, 720))
    cv2.imshow('show', img)
    cv2.waitKey(0)

def denormalize_box(box, image_size):
    width, height = image_size
    cx, cy, w, h = box
    bbox = torch.tensor([cx*width, cy*height, w*width, h*height])
    bbox = box_convert(bbox, 'cxcywh', 'xyxy')
    return bbox

