import argparse
import torch
import logging
from torchsummary import summary
from cnn_model import Net
from data_loader import LoadData
from torch.optim.lr_scheduler import StepLR
from torch import optim


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight-decay', default=0.0001, type=float,
                        help='parameter to decay weights')
    parser.add_argument('--batch-size', default=128, type=int,
                        help='size of each batch of cifar-10 training images')
    parser.add_argument('--dropout', default=0.09, help='whether to use dropout in network')
    parser.add_argument('--epochs', default=20, help='Number of Epochs')
    parser.add_argument('--lr', default=0.01, help='Default LR set to one')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse()

    # is cuda available?
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # load the data
    train_loader, test_loader = LoadData().load_data()

    model = Net(dropout=args.dropout).to(device)
    logging.debug("Model Summary {}".format(summary(model, input_size=(3, 32, 32))))

    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, nesterov=False)
    # scheduler = StepLR(optimizer, step_size=8, gamma=0.1)
    for epoch in range(args.epochs):
        # print('Epoch:', epoch+1,'LR:', scheduler.get_lr()[0])
        model.train(model, device, train_loader, optimizer)
        model.test(model, device, test_loader)
