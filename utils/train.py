import argparse

import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import numpy as np
from tensorboardX import SummaryWriter
from torch.utils.data import SubsetRandomSampler

from models import ResNet, MLP, RandomForest
from utils.dataloader import get_dataloader
from utils.new.params import root_params
from utils.new.funcs.train import preds_to_lab
from utils.new.generators import t_data_gen, train_data_gen


def load_data():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                             download=True, transform=transform)
    valid_size = 0.1
    num_train = len(train_set)
    indices = list(range(num_train))
    np.random.seed(3)
    np.random.shuffle(indices)
    split = int(np.floor(valid_size * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=128,
                                               sampler=train_sampler, num_workers=2)
    val_loader = torch.utils.data.DataLoader(train_set, batch_size=128,
                                             sampler=valid_sampler, num_workers=2)
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=128,
                                              shuffle=False, num_workers=2)
    return train_loader, val_loader, test_loader


def train_model(model, loss_criterion, train_loader, optimizer, scheduler, num_epochs, tensorboard_writer=None,
                silent=False, val_loader=None):
    for epoch in range(num_epochs):
        running_loss = 0.0
        iteration = 0
        scheduler.step()
        for iter_in_epoch, data in enumerate(train_loader, 0):
            inputs, labels = data
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = loss_criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if iteration % 100 == 99:
                train_acc = t_model(model, train_loader)
                val_acc = t_model(model, val_loader)
                av_loss = running_loss / 100
                if tensorboard_writer:
                    write_results(tensorboard_writer, av_loss, iteration, model, train_acc, val_acc)
                if not silent:
                    print_results(iter_in_epoch, epoch, av_loss, train_acc, val_acc)
                running_loss = 0
            iteration += 1
    print('Finished Training')


def print_results(iter, epoch, loss, train_acc, val_acc):
    print('[%d, %5d] loss: %.3f train_acc: %.3f, val_acc: %.3f' %
          (epoch + 1, iter + 1, loss, train_acc, val_acc))


def write_results(tensorboard_writer, loss, iter, model, train_acc, test_acc):
    tensorboard_writer.add_scalar('data/loss ', loss, iter)
    tensorboard_writer.add_scalar('data/train_acc ', test_acc, iter)
    tensorboard_writer.add_scalar('data/test_acc ', test_acc, iter)
    for name, param in model.named_parameters():
        tensorboard_writer.add_histogram(name, param.clone().cpu().data.numpy(),
                                         iter)


def t_model(model, test_loader, print_results=False):
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
    if print_results:
        print("Test acc: ", acc)
    return acc


def train(model_name, lr, num_epochs, weight_decay):
    train_loader, val_loader, test_loader = load_data()
    model = get_model_by_name(model_name)
    if model:
        writer = SummaryWriter('logs/' + model_name)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
        train_model(model, criterion, train_loader, optimizer, scheduler, num_epochs=num_epochs,
                    tensorboard_writer=writer,
                    val_loader=val_loader)
        t_model(model, test_loader, print_results=True)


def t(model, songs_list, audio_root, params, save_path):
    param, _, _, _, category = params()
    for song_name, X in t_data_gen(songs_list, audio_root, param):
        y = model.predict(X)
        preds_to_lab(y, param['hop_size'], param['fs'], category, save_path, song_name)


def createParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', )
    parser.add_argument('--train_list_path')
    parser.add_argument('--test_list_path')
    parser.add_argument('--audio_path')
    parser.add_argument('--gt_path')
    parser.add_argument('--num_epochs', default=2, type=int)
    parser.add_argument('--learning_rate', default=0.001, type=float)
    parser.add_argument('--weight_decay', default=1e-5, type=float)
    return parser


def get_model_by_name(model_name):
    if model_name.lower() == "resnet":
        return ResNet()
    elif model_name.lower() == "mlp":
        return MLP()
    elif model_name.lower() == "rf":
        return RandomForest()


def train_rf(data_path):
    rf = RandomForest()
    train_loader, val_loader = get_dataloader(data_path)
    for data in train_loader:
        inputs, labels = data
        rf.fit(inputs, labels)
    val_rf(rf, val_loader)
    return rf


def val_rf(model, val_loader, print_results=False):
    total, correct = 0, 0
    for data in val_loader:
        inputs, labels = data
        predicted = model.predict(inputs)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    acc = 100 * correct / total
    print("Test acc: ", acc)
    return acc


# songs_list='C:/Users/Daniil/Documents/Git/baseline/tangkk-mirex-ace-master/tracklists/TheBeatles1List',
# audio_root='C:/Users/Daniil/Documents/Git/baseline/tangkk-mirex-ace-master/audio/',
# gt_root='C:/Users/Daniil/Documents/Git/vkr/data/gt/'


if __name__ == '__main__':
    # parser = createParser()
    # args = parser.parse_args(sys.argv[1:])
    songs_list = 'C:/Users/Daniil/Documents/Git/baseline/tangkk-mirex-ace-master/tracklists/TheBeatles1List'
    audio_root = 'C:/Users/Daniil/Documents/Git/baseline/tangkk-mirex-ace-master/audio/'
    gt_root = 'C:/Users/Daniil/Documents/Git/vkr/data/gt/'
    save_train_data_as = 'C:/Users/Daniil/Documents/Git/vkr/data/gt/train.csv'
    # prepare train dataset
    train_data_gen(songs_list, audio_root, gt_root, root_params, save_train_data_as)
    model = train_rf(save_train_data_as)
    #val_rf(model)


    #train(args.model, args.learning_rate, args.num_epochs, args.weight_decay)
