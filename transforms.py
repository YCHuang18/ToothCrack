import random
from torchvision.transforms import functional as F
import torchvision.transforms as T


class Compose(object):

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ToTensor(object):

    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target


class RandomHorizontalFlip(object):

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)
            bbox = target["boxes"]
            # bbox: xmin, ymin, xmax, ymax
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
            target["boxes"] = bbox
            if "masks" in target:
                target["masks"] = target["masks"].flip(-1)
        return image, target


class Resize(object):

    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        # Resize image
        image = F.resize(image, self.size)

        # Resize bounding boxes
        w, h = image.shape[-1], image.shape[-2]
        boxes = target["boxes"]
        boxes[:, [0, 2]] = boxes[:, [0, 2]] * self.size[1] / w
        boxes[:, [1, 3]] = boxes[:, [1, 3]] * self.size[0] / h
        target["boxes"] = boxes

        # Resize masks if they exist
        if "masks" in target:
            masks = target["masks"]
            masks = T.Resize(self.size)(masks)
            target["masks"] = masks

        return image, target
