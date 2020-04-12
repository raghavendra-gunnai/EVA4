from torchvision import transforms
import torch, torchvision
from albumentations import (
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, RandomBrightnessContrast, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, Flip, OneOf, Compose, Normalize, Cutout
)
from albumentations.pytorch.transforms import ToTensor
from PIL import Image
import numpy as np
import albumentations as A
import albumentations.pytorch as AP


class LoadData:
    def __init__(self):
        pass
      
    class AlbumTransformer(object):
        def strong_aug(self,p=.5):
            return Compose([
                Transpose(),
                ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
                HueSaturationValue(p=0.3),
                HorizontalFlip(always_apply=False, p=0.5),
                Cutout(num_holes=1, max_h_size=8, max_w_size=8, fill_value=0.5*255)
        #        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        #        ToTensor()
              ], p=p)

        def augment(self, fun, image):
            return fun(image=image)['image']

        def __call__(self, img):
            aug = self.strong_aug(p=0.9)
            return Image.fromarray(self.augment(aug, np.array(img)))

    class TestAlbTransforms:
        def __init__(self, transforms_list=[]):
            transforms_list = []
            transforms_list.append(A.Normalize(mean=0.5,std=0.5))
            transforms_list.append(AP.ToTensor())
            self.transforms = A.Compose(transforms_list)

        def __call__(self, img):
            img = np.array(img)
            #print(img)
            return self.transforms(image=img)['image']

    def load_data(self,batch_size=128):
        transform = transforms.Compose(
            [
             self.AlbumTransformer(),
             transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
             #transforms.RandomErasing(p=0.5, scale=(0.005,0.055), ratio=(0.05,0.5))
             ])
        
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=transform)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                   shuffle=True, num_workers=4)
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                               download=True, transform=transform)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                  shuffle=False, num_workers=4)

        # classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

        return train_loader, test_loader