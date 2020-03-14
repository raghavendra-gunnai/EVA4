from torchvision import transforms
import torch, torchvision


class LoadData:
    def __init__(self):
        pass

    def load_data(self):
        transform = transforms.Compose(
            [
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
             transforms.RandomErasing(p=0.5, scale=(0.005,0.055), ratio=(0.05,0.5))])
        
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=transform)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                                   shuffle=True, num_workers=4)
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                               download=True, transform=transform)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=128,
                                                  shuffle=False, num_workers=4)

        # classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

        return train_loader, test_loader
