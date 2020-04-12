import torch.optim as optim
from tqdm import tqdm_notebook, tnrange
import torch.nn.functional as F
class lrRangeFinder():
  def __init__(self,model,dataloader,device):
    self.model=model
    self.dataloader=dataloader
    self.learning_rates=[]
    self.device = device
    self.training_accuracy=[]
    self.learning_rate=0.00001
    self.average_accuracy=0.0
  def plot(self,epochs):
    for i in range(epochs):
        print("starting {} epoch:".format(i+1))
        self.model.train()
        self.average_accuracy=0.0
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9, nesterov=True)
        pbar = tqdm_notebook(self.dataloader)
        for data, target in pbar:
            # get samples
            data, target = data.to(self.device), target.to(self.device)

            # Init
            self.optimizer.zero_grad()
            # In PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes. 
            #  Because of this, when you start your training loop, ideally you should zero out the gradients so that you do the parameter update correctly.
            # Predict
            y_pred = self.model(data)

            # Calculate loss
            loss = F.nll_loss(y_pred, target)
            # Backpropagation
            loss.backward()
            # Update pbar-tqdm
            pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct = pred.eq(target.view_as(pred)).sum().item()
            self.average_accuracy+=(correct/(len(data)))

        self.learning_rates.append(self.learning_rate)
        self.training_accuracy.append(self.average_accuracy/len(data))
        self.learning_rate*=10
    return self.learning_rates,self.training_accuracy        