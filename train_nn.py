import argparse

import torch
import torch.optim as optim
import torch.nn as nn
from tensorboardX import SummaryWriter

from models import LSTMClassifier
from dataloader import get_train_val_seq_dataloader
from preprocess.chords import preds_to_lab
from preprocess.generators import gen_test_data, gen_train_data
from preprocess.params import root_params
import sys

torch.set_default_dtype(torch.float64)
use_gpu = torch.cuda.is_available()


def train_model(model, loss_criterion, train_loader, optimizer, scheduler, num_epochs, tensorboard_writer=None,
                silent=False, val_loader=None, test_every=10):
    for epoch in range(num_epochs):
        running_loss = 0.0
        iteration = 0
        for iter_in_epoch, data in enumerate(train_loader, 0):
            inputs, labels = data
            if use_gpu:
                inputs, labels = inputs.cuda(), labels.cuda()
            model.zero_grad()
            outputs = model(inputs)
            outputs = outputs.view(-1, outputs.size(2))
            labels = labels.view(-1)
            loss = loss_criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if iteration % test_every == test_every - 1:
                # print statistics
                train_acc = val_model(model, train_loader)
                val_acc = val_model(model, val_loader)
                av_loss = running_loss / test_every
                if tensorboard_writer:
                    write_results(tensorboard_writer, av_loss, iteration, model, train_acc, val_acc)
                if not silent:
                    print_results(iter_in_epoch, epoch, av_loss, train_acc, val_acc)
                running_loss = 0
            iteration += 1
            scheduler.step()
    print('Finished Training')
    val_model(model, val_loader, print_results=True)


def print_results(iter, epoch, loss, train_acc, val_acc):
    print('[%d, %5d] loss: %f train_acc: %.3f, val_acc: %.3f' %
          (epoch + 1, iter + 1, loss, train_acc, val_acc))


def write_results(tensorboard_writer, loss, iter, model, train_acc, test_acc):
    tensorboard_writer.add_scalar('data/loss ', loss, iter)
    tensorboard_writer.add_scalar('data/train_acc ', test_acc, iter)
    tensorboard_writer.add_scalar('data/test_acc ', test_acc, iter)
    for name, param in model.named_parameters():
        tensorboard_writer.add_histogram(name, param.clone().cpu().data.numpy(),
                                         iter)


def val_model(model, test_loader, print_results=False):
    correct, total, acc = 0, 0, 0
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            if use_gpu:
                inputs, labels = inputs.cuda(), labels.cuda()
            outputs = model(inputs)
            predicted = outputs.topk(1, dim=2)[1].squeeze()
            total += labels.size(0) * labels.size(1)
            correct += (predicted == labels).sum().item()
    if total:
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
    parser.add_argument('-m', '--model', required=True)
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
    parser.add_argument('--hidden_dim', default=50, type=int)
    parser.add_argument('--num_layers', default=2, type=int)
    parser.add_argument('--batch_size', default=40, type=int)
    parser.add_argument('--test_every', default=10, type=int)
    parser.add_argument('--use_librosa', default=True, type=bool)
    parser.add_argument('--save_model_as', type=str)
    return parser


def train_LSTM(model, train_path, num_epochs, weight_decay, lr, batch_size=4, test_every=10):
    if use_gpu:
        model = model.cuda()
    conv_root = args.conv_root
    if args.use_librosa:
        conv_root= conv_root + '/librosa/'
    else:
        conv_root = conv_root + '/mauch/'
    train_loader, val_loader = get_train_val_seq_dataloader(train_path, batch_size)
    writer = SummaryWriter('logs/' + 'LSTM')
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    train_model(model, criterion, train_loader, optimizer, scheduler, num_epochs=num_epochs,
                tensorboard_writer=writer,
                val_loader=val_loader, test_every=test_every)
    return model


def get_params_by_category(category):
    if category == 'MirexRoot':
        _, _, _, _, _, y_size = root_params()
        return root_params, y_size


def save_model(model, name):
    import _pickle as pickle
    output = open(f'pretrained/{name}.pkl', 'wb')
    pickle.dump(model, output, -1)
    output.close()


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
    model = LSTMClassifier(input_size=84, hidden_dim=args.hidden_dim, output_size=y_size, num_layers=args.num_layers,
                           use_gpu=use_gpu)
    model = train_LSTM(model, train_path=conv_list, num_epochs=args.num_epochs,
                       weight_decay=args.weight_decay, lr=args.learning_rate, batch_size=args.batch_size,
                       test_every=args.test_every)
    if args.save_model_as:
        save_model(model, args.save_model_as)
