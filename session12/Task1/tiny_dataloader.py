import torch
import torchvision
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
import numpy as np
import tiny_imagenet

import cv2
from albumentations import Compose
from albumentations.pytorch import ToTensor
from albumentations import (RandomCrop, Normalize, HorizontalFlip, Flip, 
                            Rotate, RGBShift, GaussNoise, PadIfNeeded, Cutout, CoarseDropout)
class Alb_Transform_Train:
    def __init__(self):
        self.alb_transform = Compose([
            Rotate((-30.0, 30.0)),
            HorizontalFlip(),
            RGBShift(r_shift_limit=50, g_shift_limit=50, b_shift_limit=50, p=0.5),
            Cutout(num_holes=8, max_h_size=8, max_w_size=8, fill_value=[0.4914*255, 0.4822*255, 0.4465*255], always_apply=False, p=0.7),
            Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ToTensor()
        ])
    
    def __call__(self, img):
        img = np.array(img)
        img = self.alb_transform(image=img)['image']
        return img

class Alb_Transform_Test:
    def __init__(self):
        self.alb_transform = Compose([
            Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ToTensor()
        ])
    
    def __call__(self, img):
        img = np.array(img)
        img = self.alb_transform(image=img)['image']
        return img

def get_train_test_data(path):
  g_classes, g_class_to_idx = tiny_imagenet.find_classes(path)
  train_file_names = []
  test_file_names = []
  train_classname_dict = dict()    
  test_classname_dict = dict()    
  classes_dict = dict()

  train_file_names, test_file_names, train_classname_dict, test_classname_dict, classes_dict = tiny_imagenet.split_data(path,g_class_to_idx)
  
  train = tiny_imagenet.TinyImagenet200(path,train_file_names, train_classname_dict,classes_dict,transform = Alb_Transform_Train())
  test = tiny_imagenet.TinyImagenet200(path,test_file_names,test_classname_dict,classes_dict,Alb_Transform_Test())
  return train, test

def get_train_test_loaders(train, test, batch_size_train, batch_size_test=100, batch_size_noncuda=64):
  SEED = 1

  # CUDA?
  cuda = torch.cuda.is_available()
  print("CUDA Available?", cuda)

  # For reproducibility
  torch.manual_seed(SEED)

  if cuda:
    torch.cuda.manual_seed(SEED)

  # dataloader arguments - something you'll fetch these from cmdprmt
  trainloader_args = dict(shuffle=True, batch_size=batch_size_train, num_workers=2, pin_memory=True) if cuda else dict(shuffle=True, batch_size=batch_size_noncuda)
  testloader_args = dict(shuffle=False, batch_size=batch_size_test, num_workers=2, pin_memory=True) if cuda else dict(shuffle=False, batch_size=batch_size_noncuda)


  # train dataloader
  train_loader = torch.utils.data.DataLoader(train, **trainloader_args)

  # test dataloader
  test_loader = torch.utils.data.DataLoader(test, **testloader_args)

  return train_loader, test_loader

def display(train_loader, num_imgs=5):
  # functions to show an image
  def imshow(img):
      img = img / 2 + 0.5     # unnormalize
      npimg = img.numpy()
      plt.imshow(np.transpose(npimg, (1, 2, 0)))

  # get some random training images
  dataiter = iter(train_loader)
  images, labels = dataiter.next()

  # show images
  imshow(torchvision.utils.make_grid(images[:num_imgs]))
  #imshow(images[1])
  #print(classes[labels[1]])
  #print(labels)
  #print(' '.join('%5s' % classes[labels[j]] for j in range(4)))