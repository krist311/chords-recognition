from models import RandomForest
from dataloader import get_train_val_rf_dataloader
from preprocess.chords import preds_to_lab
from preprocess.generators import gen_test_data, gen_train_data
import sys
import glog as log
from pprint import pformat

from utils.parser import get_train_rf_parser
from utils.utils import get_params_by_category


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
    val_acc = val_rf(model, val_loader, )
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
        print("Val acc: ", acc)
    return acc


def print_results(iter, train_acc, val_acc):
    print('[%5d] train_acc: %.3f, val_acc: %.3f' %
          (iter + 1, train_acc, val_acc))


def t(model, songs_list, audio_root, params, save_path):
    param, _, _, _, category, _ = params()
    for song_name, X in gen_test_data(songs_list, audio_root, param):
        y = model.predict(X)
        preds_to_lab(y, param['hop_size'], param['fs'], category, save_path, song_name)


if __name__ == '__main__':
    parser = get_train_rf_parser()
    args = parser.parse_args(sys.argv[1:])
    log.info('Arguments:\n' + pformat(args.__dict__))
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
