import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
from tensorboardX import SummaryWriter

from models import MLP, ResNet


def load_data():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                             shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    return trainloader, testloader, classes


def train_model(model, loss_criterion, train_loader, optimizer, num_epochs, tensorboard_writer=None, model_name=None,
                silent=False, test_loader=None):
    for epoch in range(num_epochs):

        running_loss = 0.0
        for iter, data in enumerate(train_loader, 0):
            inputs, labels = data

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = loss_criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            current_loss = loss.item()
            if tensorboard_writer and iter % 2000 == 1999:
                test_model(model, test_loader, iter=12500 * epoch + iter, model_name=model_name,
                           tensorboard_writer=tensorboard_writer)
            if tensorboard_writer and iter % 10 == 0:
                tensorboard_writer.add_scalar('data/loss ' + model_name, loss.item(), 12500 * epoch + iter)
                for name, param in model.named_parameters():
                    tensorboard_writer.add_histogram(model_name + " " + name, param.clone().cpu().data.numpy(),
                                                     12500 * epoch + iter)
            if not silent:
                running_loss += current_loss
                if iter % 2000 == 1999:  # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, iter + 1, running_loss / 2000))
                    running_loss = 0.0

    print('Finished Training')


def test_model(model, test_loader, tensorboard_writer=None, iter=None, model_name=None):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = 100 * correct / total
    if (tensorboard_writer):
        tensorboard_writer.add_scalar('data/test_acc ' + model_name, acc, iter)
    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))


def main():
    train_loader, test_loader, classes = load_data()
    writer = SummaryWriter()

    # MLP
    mlp = MLP()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(mlp.parameters(), lr=0.001)
    train_model(mlp, criterion, train_loader, optimizer, num_epochs=2, model_name='MPL', tensorboard_writer=writer,
                test_loader=test_loader)
    test_model(mlp, test_loader)

    # Resnet_like
    mlp = ResNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(mlp.parameters(), lr=0.001)
    train_model(mlp, criterion, train_loader, optimizer, num_epochs=2, model_name='ResNet', tensorboard_writer=writer,
                test_loader=test_loader)
    test_model(mlp, test_loader)


if __name__ == '__main__':
    main()
