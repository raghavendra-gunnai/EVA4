from lr_finder import LRFinder
import torch.optim as optim
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class LRUtils:
	def find_lr(self,model, device, train_loader, lr_val=1e-8, decay=1e-2):
		criterion = nn.CrossEntropyLoss()
		optimizer = optim.SGD(model.parameters(), lr=lr_val, weight_decay=decay)
		lr_finder = LRFinder(model, optimizer, criterion, device)
		lr_finder.range_test(train_loader, end_lr=100, num_iter=100, step_mode="exp")
		lr_finder.plot()
		return lr_finder

	def plot_lrfinder(self,train_acc, test_acc, trainloss_, testloss_):
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

	def get_triangular_lr(self,iteration, stepsize, base_lr, max_lr):
	    cycle = np.floor(1 + iteration/(2  * stepsize))
	    x = np.abs(iteration/stepsize - 2 * cycle + 1)
	    lr = base_lr + (max_lr - base_lr) * np.maximum(0, (1-x))
	    return lr

	def get_lr_and_plot(self,num_iterations = 10000, stepsize = 1000, base_lr = 0.0001, max_lr = 0.001):
		for iteration in range(num_iterations):
			lr_trend = []
			lr = self.get_triangular_lr(iteration,stepsize,base_lr,max_lr)
			lr_trend.append(lr)
		plt.plot(lr_trend)