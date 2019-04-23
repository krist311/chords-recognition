import argparse

from models import RandomForest
from dataloader import get_train_val_rf_dataloader
from preprocess.chords import preds_to_lab
from preprocess.generators import gen_test_data, gen_train_data
from preprocess.params import root_params, maj_min_params, maj_min_bass_params, seventh_params, seventh_bass_params
import sys


def train_rf(model, data_path):
    train_loader, val_loader = get_train_val_rf_dataloader(data_path)
    for i, data in enumerate(train_loader):
        inputs, labels = data
        model.fit(inputs, labels)
        model.n_estimators += 1
        if i % 10 == 9:
            train_acc = val_rf(model, train_loader)
            val_acc = val_rf(model, val_loader)
            print_results(i, train_acc, val_acc)
    train_acc = val_rf(model, train_loader)
    val_acc = val_rf(model, val_loader)
    print_results(99, train_acc, val_acc)
    return model


def val_rf(model, val_loader, print_results=False):
    total, correct = 0, 0
    for data in val_loader:
        inputs, labels = data
        predicted = model.predict(inputs)
        total += labels.size(0)
        correct += (predicted == labels.numpy()).sum().item()
    acc = 100 * correct / total
    if print_results:
        print("Test acc: ", acc)
    return acc


def print_results(iter, train_acc, val_acc):
    print('[%5d] train_acc: %.3f, val_acc: %.3f' %
          (iter + 1, train_acc, val_acc))


def t(model, songs_list, audio_root, params, save_path):
    param, _, _, _, category = params()
    for song_name, X in gen_test_data(songs_list, audio_root, param):
        y = model.predict(X)
        preds_to_lab(y, param['hop_size'], param['fs'], category, save_path, song_name)


def get_params_by_category(category):
    params, y_size = 0, 0
    if category == 'MirexRoot':
        params = root_params
    elif category == 'MirexMajMin':
        params = maj_min_params
    elif category == 'MirexMajMinBass':
        params = maj_min_bass_params
    elif category == 'MirexSevenths':
        params = seventh_params
    elif category == 'MirexSeventhsBass':
        params = seventh_bass_params
    _, _, _, _, _, y_size = params()
    return params, y_size


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
    parser.add_argument('--criterion', default='entropy', type=str)
    parser.add_argument('--max_features', default='log2', type=str)
    parser.add_argument('--n_estimators', default='1', type=int)
    parser.add_argument('--use_librosa', default=True, type=bool)
    return parser


if __name__ == '__main__':
    parser = createParser()
    args = parser.parse_args(sys.argv[1:])
    # prepare train dataset
    params, y_size = get_params_by_category(args.category)
    conv_root = args.conv_root
    if args.use_librosa:
        conv_root = conv_root + '/librosa/'
    else:
        conv_root = conv_root + '/mauch/'
    conv_list = args.conv_list
    if not conv_list:
        conv_list = gen_train_data(args.songs_list, args.audio_root, args.gt_root, params, conv_root,
                                   args.subsong_len, args.song_len, use_librosa=args.use_librosa)
    model = RandomForest(criterion=args.criterion, max_features=args.max_features, n_estimators=args.n_estimators)
    model = train_rf(model, conv_list)
