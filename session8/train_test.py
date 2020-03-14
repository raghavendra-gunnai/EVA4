from tqdm import tqdm
import torch.nn as nn
import torch
import torch.nn.functional as F


class Trainer:
  def __init__(self):
    self.train_losses = []
    self.train_acc = []
    self.test_losses = []
    self.test_acc = []

  def train(self, model, device, train_loader, optimizer, loss_fun=None):
    model.train()
    pbar = tqdm(train_loader)
    correct = 0
    processed = 0
    for batch_idx, (data, target) in enumerate(pbar):
      data, target = data.to(device), target.to(device)

      optimizer.zero_grad()
      y_pred = model(data)

      # Calculate loss
      if loss_fun is None:
        loss = F.nll_loss(y_pred, target)
      else:
        loss = loss_fun(y_pred, target)
      self.train_losses.append(loss)

      # Backpropagation
      loss.backward()
      optimizer.step()

      pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
      correct += pred.eq(target.view_as(pred)).sum().item()
      processed += len(data)

      pbar.set_description(desc=f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
      self.train_acc.append(100*correct/processed)

  def test(self, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
      for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        test_loss += F.nll_loss(output, target, reduction='sum').item() #test_loss += (pred != target).sum().item()
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    self.test_losses.append(test_loss)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
      test_loss, correct, len(test_loader.dataset),
      100. * correct / len(test_loader.dataset)))

    self.test_acc.append(100. * correct / len(test_loader.dataset))