import numpy as np
import cv2
import io
import os
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from torchvision import transforms
import zipfile
import torchvision.datasets.folder
from PIL import Image

import random
import math

class TinyImagenet200(Dataset):
  def __init__(self, root, files, classname_dict, class_dict, transform=ToTensor()):
    self.m_root = root
    self.m_class_dict = class_dict
    self.m_classname_dict = classname_dict
    #self.m_file_path_dict = file_path_dict
    self.m_files = files
    self.m_transform = transform

  def __len__(self):
    return len(self.m_files)

  def __getitem__(self, index):
    file_name = self.m_files[index]
    class_name = self.m_classname_dict[file_name]
    dir_path = os.path.join(self.m_root, class_name)
    image_dir = os.path.join(dir_path, 'images')
    image_name = os.path.join(image_dir,file_name)
    image = Image.open(image_name).convert('RGB') 
    try:
      if self.m_transform:
        image = self.m_transform(image)
    except:
      print('Error while transform:', image_name, image.shape)
    return image, self.m_class_dict[file_name]

def split_data(dir,g_class_to_idx):
    
    train_file_names = []
    train_class_names = []
    test_file_names = []
    test_class_names = []
    classes_dict = dict()
    test_classname_dict = dict()
    train_classname_dict = dict()

    dire_name = ''
    idx =0
    for root, dirs, fnames in sorted(os.walk(dir)):
      idx = 0
      for fname in fnames:
        if is_image_file(fname):
          idx += 1
          path = os.path.join(root, fname)
          dir_name = os.path.basename(os.path.dirname(root))
          classes_dict[fname] = g_class_to_idx[dir_name]

          if idx > 7:
            test_file_names.append(fname)
            test_classname_dict[fname] = os.path.basename(os.path.dirname(root))
          else:
            train_file_names.append(fname)
            train_classname_dict[fname] = os.path.basename(os.path.dirname(root))

          idx = idx % 10
    return  train_file_names, test_file_names, train_classname_dict, test_classname_dict, classes_dict


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx
