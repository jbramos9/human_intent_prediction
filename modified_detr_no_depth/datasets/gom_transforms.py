import torchvision
import torch
import random
import PIL
import torchvision.transforms as T
import torchvision.transforms.functional as F
from util.box_ops import box_xyxy_to_cxcywh


def hflip(image, target, image_set='train'):
    flipped_image = F.hflip(image)
    target = target.copy()
    if image_set in ['test']:
        return flipped_image, target

    w, h = image.size
    if "boxes" in target:
        boxes = target["boxes"]
        boxes = boxes[:, [2, 1, 0, 3]] * \
            torch.as_tensor([-1, 1, -1, 1]) + torch.as_tensor([w, 0, w, 0])
        target["boxes"] = boxes

    return flipped_image, target


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target, image_set='train'):
        if random.random() < self.p:
            return hflip(img, target, image_set)
        return img, target


class RandomAdjustImage(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target, image_set='train'):
        if random.random() < self.p:
            img = F.adjust_brightness(
                img, random.choice([0.8, 0.9, 1.0, 1.1, 1.2]))
        if random.random() < self.p:
            img = F.adjust_contrast(
                img, random.choice([0.8, 0.9, 1.0, 1.1, 1.2]))
        return img, target


class RandomSelect(object):
    """
    Randomly selects between transforms1 and transforms2,
    with probability p for transforms1 and (1 - p) for transforms2
    """

    def __init__(self, transforms1, transforms2, p=0.5):
        self.transforms1 = transforms1
        self.transforms2 = transforms2
        self.p = p

    def __call__(self, img, target, image_set='train'):
        if random.random() < self.p:
            return self.transforms1(img, target, image_set)
        return self.transforms2(img, target, image_set)


def resize(image, target, size, max_size=None, image_set='train'):
    def get_size_with_aspect_ratio(image_size, size, max_size=None):
        w, h = image_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(
                    round(max_size * min_original_size / max_original_size))
        if (w <= h and w == size) or (h <= w and h == size):
            return h, w
        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)
        return oh, ow

    rescale_size = get_size_with_aspect_ratio(
        image_size=image.size, size=size, max_size=max_size)
    rescaled_image = F.resize(image, rescale_size)

    if target is None:
        return rescaled_image, None
    target = target.copy()
    if image_set in ['test']:
        return rescaled_image, target

    ratios = tuple(float(s) / float(s_orig)
                   for s, s_orig in zip(rescaled_image.size, image.size))
    ratio_width, ratio_height = ratios

    if "boxes" in target:
        boxes = target["boxes"]
        scaled_boxes = boxes * \
            torch.as_tensor([ratio_width, ratio_height,
                            ratio_width, ratio_height])
        target["boxes"] = scaled_boxes
    return rescaled_image, target


class RandomResize(object):
    def __init__(self, sizes, max_size=None):
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes
        self.max_size = max_size

    def __call__(self, img, target=None, image_set='train'):
        size = random.choice(self.sizes)
        return resize(img, target, size, self.max_size, image_set)


def crop(image, org_target, region, image_set='train'):
    cropped_image = F.crop(image, *region)
    target = org_target.copy()
    if image_set in ['test']:
        return cropped_image, target

    i, j, h, w = region
    fields = ["objects"]

    if "boxes" in target:
        boxes = target["boxes"]
        max_size = torch.as_tensor([w, h], dtype=torch.float32)
        cropped_boxes = boxes - torch.as_tensor([j, i, j, i])
        cropped_boxes = torch.min(cropped_boxes.reshape(-1, 2, 2), max_size)
        cropped_boxes = cropped_boxes.clamp(min=0)
        target["boxes"] = cropped_boxes.reshape(-1, 4)
        fields.append("boxes")

    # remove elements for which the boxes or masks that have zero area
    if "boxes" in target:
        cropped_boxes = target['boxes'].reshape(-1, 2, 2)
        keep = torch.all(cropped_boxes[:, 1, :]
                          > cropped_boxes[:, 0, :], dim=1)
        if keep.any().sum() == 0:
            return image, org_target
        for field in fields:
            target[field] = target[field][keep]
    return cropped_image, target


class RandomSizeCrop(object):
    def __init__(self, min_size: int, max_size: int):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, img: PIL.Image.Image, target: dict, image_set='train'):
        w = random.randint(self.min_size, min(img.width, self.max_size))
        h = random.randint(self.min_size, min(img.height, self.max_size))
        region = T.RandomCrop.get_params(img, (h, w))
        return crop(img, target, region, image_set)


class ToTensor(object):
    def __call__(self, img, target, image_set='train'):
        return torchvision.transforms.functional.to_tensor(img), target


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target, image_set='train'):
        image = torchvision.transforms.functional.normalize(
            image, mean=self.mean, std=self.std)
        if target is None:
            return image, None
        target = target.copy()
        # if image_set in ['test']: #need to normalize the test set
        #     return image, target
        h, w = image.shape[-2:]

        if "boxes" in target:
            boxes = target["boxes"]
            boxes = box_xyxy_to_cxcywh(boxes)
            boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
            target["boxes"] = boxes

        return image, target


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target, image_set='train'):
        for t in self.transforms:
            image, target = t(image, target, image_set)
        return image, target


def make_gom_transforms(image_set, test_scale=-1):
    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
    normalize = Compose([
        ToTensor(),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    if image_set == 'train':
        return Compose([
            RandomHorizontalFlip(),
            RandomAdjustImage(),
            RandomSelect(
                RandomResize(scales),
                Compose([
                    RandomResize([400, 500, 600]),
                    # RandomSizeCrop(400, 600),
                    RandomResize(scales),
                ])
            ),
            normalize,
        ])
    if image_set == 'test':
        if test_scale == -1:
            return Compose([
                normalize,
            ])
        assert 400 <= test_scale <= 800, test_scale
        return Compose([
            RandomResize([test_scale], max_size=1333),
            normalize,
        ])
    raise ValueError(f'unknown {image_set}')
