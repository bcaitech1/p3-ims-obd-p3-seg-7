import numpy as np
import cv2

from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates

import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations.core.transforms_interface import ImageOnlyTransform, DualTransform


class BasicAugmentation:
    def __init__(self):
        self.transform = A.Compose([
            ToTensorV2()
        ])

    def __call__(self, **data):
        return self.transform(**data)


class ImagenetDefaultAugmentation:
    def __init__(self):
        self.transform = A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

    def __call__(self, **data):
        return self.transform(**data)


class CustomAugmentation:
    def __init__(self):
        self.transform = A.Compose([
            A.HorizontalFlip(0.5),
            A.VerticalFlip(0.5),
            A.Rotate(),
            # A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

    def __call__(self, **data):
        return self.transform(**data)

class CustomElasticAugmentation:
    def __init__(self, alpha=512*2, sigma=512*0.08, alpha_affine=512*0.08):
        self.transform = A.Compose([
            A.HorizontalFlip(0.5),
            A.VerticalFlip(0.5),
            A.Rotate(),
            ElasticTransform(alpha, sigma, alpha_affine, p=1),
            # A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

    def __call__(self, **data):
        return self.transform(**data)


def elastic_transform(image, alpha, sigma, alpha_affine, seed, **params):
    """Elastic deformation of images as described in [Simard2003]_ (with modifications).
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
         Convolutional Neural Networks applied to Visual Document Analysis", in
         Proc. of the International Conference on Document Analysis and
         Recognition, 2003.

     Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
    """

    random_state = np.random.RandomState(seed)

    shape = image.shape
    shape_size = shape[:2]
    
    # Random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32([center_square + square_size, [center_square[0]+square_size, center_square[1]-square_size], center_square - square_size])
    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dz = np.zeros_like(dx)

    if len(shape) != 2:
        x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
        indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)), np.reshape(z, (-1, 1))

    else:
        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))

    return map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)

class ElasticTransform(DualTransform):
    def __init__(self, alpha, sigma, alpha_affine, p=1):
        super(ElasticTransform, self).__init__(p=p)
        self.alpha = alpha
        self.sigma = sigma
        self.alpha_affine = alpha_affine

    def apply(self, image, **params):
        return elastic_transform(image, self.alpha, self.sigma, self.alpha_affine, **params)

    def get_params(self):
        return {'seed' : np.random.randint(999999)}