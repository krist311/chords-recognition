import argparse

import torch
import torch.optim as optim
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch.autograd import Variable

from models import LSTMClassifier
from dataloader import get_train_val_seq_dataloader
from preprocess.chords import preds_to_lab
from preprocess.generators import gen_test_data, gen_train_data
from preprocess.params import root_params
import sys

torch.set_default_dtype(torch.float64)
use_gpu = torch.cuda.is_available()


def train_model(model, loss_criterion, train_loader, optimizer, scheduler, num_epochs, tensorboard_writer=None,
                silent=False, val_loader=None):
    for epoch in range(num_epochs):
        running_loss = 0.0
        iteration = 0
        scheduler.step()
        for iter_in_epoch, data in enumerate(train_loader, 0):
            inputs, labels = data
            if use_gpu:
                inputs, labels = Variable(inputs.cuda()), labels.cuda()
            else:
                inputs = Variable(inputs)
            model.zero_grad()
            outputs = model(inputs)
            loss = loss_criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            if iteration % 10 == 9:
            # print statistics
                running_loss += loss.item()
                train_acc = val_model(model, train_loader)
                val_acc = val_model(model, val_loader)
                av_loss = running_loss/10
                if tensorboard_writer:
                    write_results(tensorboard_writer, av_loss, iteration, model, train_acc, val_acc)
                if not silent:
                    print_results(iter_in_epoch, epoch, av_loss, train_acc, val_acc)
                running_loss = 0
            iteration += 1
    print('Finished Training')
    val_model(model, val_loader, print_results=True)


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


def val_model(model, test_loader, print_results=False):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)*labels.size(1)
            correct += (predicted == labels).sum().item()
    acc = 100 * correct / total
    if print_results:
        print("Val acc: ", acc)
    return acc


def t(model, songs_list, audio_root, params, save_path):
    param, _, _, _, category = params()
    for song_name, X in gen_test_data(songs_list, audio_root, param):
        y = model.predict(X)
        preds_to_lab(y, param['hop_size'], param['fs'], category, save_path, song_name)


def createParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', )
    parser.add_argument('--num_epochs', default=2, type=int)
    parser.add_argument('--learning_rate', default=0.01, type=float)
    parser.add_argument('--weight_decay', default=1e-5, type=float)
    parser.add_argument('--songs_list', default='data/tracklists/TheBeatles180List', type=str)
    parser.add_argument('--audio_root', default='data/audio/', type=str)
    parser.add_argument('--gt_root', default='data/gt/', type=str)
    parser.add_argument('--conv_root', default='data/converted/', type=str)
    parser.add_argument('--conv_list', default='', type=str)
    parser.add_argument('--category', default='MirexRoot', type=str)
    parser.add_argument('--subsong_len', default=40, type=int)
    parser.add_argument('--song_len', default=180, type=int)
    parser.add_argument('--hidden_dim', default=200, type=int)
    parser.add_argument('--num_layers', default=2, type=int)
    return parser


def train_LSTM(model, train_path, num_epochs, weight_decay, lr):
    if use_gpu:
        model = model.cuda()
    train_loader, val_loader = get_train_val_seq_dataloader(train_path)
    writer = SummaryWriter('logs/' + 'LSTM')
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    train_model(model, criterion, train_loader, optimizer, scheduler, num_epochs=num_epochs,
                tensorboard_writer=writer,
                val_loader=val_loader)
    return model


def get_params_by_category(category):
    if category == 'MirexRoot':
        _, _, _, _, _, y_size = root_params()
        return root_params, y_size


if __name__ == '__main__':
    use_gpu = torch.cuda.is_available()
    parser = createParser()
    args = parser.parse_args(sys.argv[1:])
    # prepare train dataset
    params, y_size = get_params_by_category(args.category)
    conv_list = args.conv_list
    if not conv_list:
        conv_list = gen_train_data(args.songs_list, args.audio_root, args.gt_root, params, args.conv_root,
                                   args.subsong_len, args.song_len)
    model = LSTMClassifier(input_size=252, hidden_dim=args.hidden_dim, output_size=y_size, num_layers=args.num_layers)
    train_LSTM(model, train_path=conv_list, num_epochs=args.num_epochs,
               weight_decay=args.weight_decay, lr=args.learning_rate)
