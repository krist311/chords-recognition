import argparse

from models import RandomForest
from dataloader import get_train_val_rf_dataloader
from preprocess.chords import preds_to_lab
from preprocess.generators import gen_test_data, gen_train_data
from preprocess.params import root_params
import sys


def t(model, songs_list, audio_root, params, save_path):
    param, _, _, _, category = params()
    for song_name, X in gen_test_data(songs_list, audio_root, param):
        y = model.predict(X)
        preds_to_lab(y, param['hop_size'], param['fs'], category, save_path, song_name)


def train_rf(data_path):
    rf = RandomForest()
    train_loader, val_loader = get_train_val_rf_dataloader(data_path)
    for i, data in enumerate(train_loader):
        inputs, labels = data
        rf.fit(inputs, labels)
        rf.n_estimators += 1
        if i % 10 == 9:
            train_acc = val_rf(rf, train_loader)
            val_acc = val_rf(rf, val_loader)
            print_results(i, train_acc, val_acc)
    return rf


def val_rf(model, val_loader, print_results=False):
    total, correct = 0, 0
    for data in val_loader:
        inputs, labels = data
        predicted = model.predict(inputs)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    acc = 100 * correct / total
    if print_results:
        print("Test acc: ", acc)
    return acc


def print_results(iter, train_acc, val_acc):
    print('[%5d] train_acc: %.3f, val_acc: %.3f' %
          (iter + 1, train_acc, val_acc))


def get_params_by_category(category):
    if category == 'MirexRoot':
        _, _, _, _, _, y_size = root_params()
        return root_params, y_size


def createParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--songs_list', default='data/tracklists/TheBeatles180List', type=str)
    parser.add_argument('--audio_root', default='data/audio/', type=str)
    parser.add_argument('--gt_root', default='data/gt/', type=str)
    parser.add_argument('--conv_root', default='data/converted/', type=str)
    parser.add_argument('--conv_list', default='', type=str)
    parser.add_argument('--category', default='MirexRoot', type=str)
    parser.add_argument('--subsong_len', default=40, type=int)
    parser.add_argument('--song_len', default=180, type=int)
    return parser


if __name__ == '__main__':
    parser = createParser()
    args = parser.parse_args(sys.argv[1:])
    # prepare train dataset
    params, y_size = get_params_by_category(args.category)
    conv_list = args.conv_list
    if not conv_list:
        conv_list = gen_train_data(args.songs_list, args.audio_root, args.gt_root, params, args.conv_root,
                                   args.subsong_len, args.song_len)
    train_rf(conv_list)
