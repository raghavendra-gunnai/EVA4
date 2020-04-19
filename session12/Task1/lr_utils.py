import matplotlib.pyplot as plt
import numpy as np
from torch_lr_finder import LRFinder
import os

# functions to show an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

def lrfinder(net, optimizer, criterion, trainloader, valloader):
  lr_finder = LRFinder(net, optimizer, criterion, device="cuda")
  lr_finder.range_test(trainloader, val_loader=valloader, end_lr=10, num_iter=100, step_mode="exponential")
  lr_finder.plot() 
  lr_finder.reset()

def plot_acc_loss(train_acc, test_acc, trainloss_, testloss_):
  fig, axs = plt.subplots(2,2,figsize=(10,10))
  axs[0,0].plot(train_acc)
  axs[0,0].set_title("Training Accuracy")
  axs[0,0].set_xlabel("Batch")
  axs[0,0].set_ylabel("Accuracy")
  axs[0,1].plot(test_acc) 
  axs[0,1].set_title("Test Accuracy")
  axs[0,1].set_xlabel("Batch")
  axs[0,1].set_ylabel("Accuracy")
  axs[1,0].plot(trainloss_)
  axs[1,0].set_title("Training Loss")
  axs[1,0].set_xlabel("Batch")
  axs[1,0].set_ylabel("Loss")
  axs[1,1].plot(testloss_) 
  axs[1,1].set_title("Test Loss")
  axs[1,1].set_xlabel("Batch")
  axs[1,1].set_ylabel("Loss")

def getclass_level():
  maps = []
  for root, dirs, files in tqdm(os.walk(os.getcwd()+'/tiny-imagenet-200/train/')):  # replace the . with your starting directory
    for file in files:  
      maps.append(file.split('_')[0])
      break
    maps.sort()
  f = open(os.getcwd()+'/tiny-imagenet-200/words.txt')
  count = 1;
  results = {}
  for line in f:
    results[line.split('\t')[0]]=line.split('\t')[1].split('\n')[0]
  f.close()
  return maps,results