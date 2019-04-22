import argparse
import _pickle as pickle
import sys

from dataloader import get_test_rf_dataloader
from preprocess.generators import gen_train_data
from preprocess.params import root_params, maj_min_params, maj_min_bass_params, seventh_params, seventh_bass_params
from train_rf import val_rf


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


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--songs_list', default='data/tracklists/TheBeatles180List', type=str)
    parser.add_argument('--audio_root', default='data/audio/', type=str)
    parser.add_argument('--gt_root', default='data/gt/', type=str)
    parser.add_argument('--conv_root', type=str)
    parser.add_argument('--conv_list', default='TheBeatles180List_converted.txt', type=str, required=True)
    parser.add_argument('--category', default='MirexRoot', type=str)
    parser.add_argument('--subsong_len', default=40, type=int)
    parser.add_argument('--song_len', default=180, type=int)
    return parser


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args(sys.argv[1:])
    model_dump = open(args.model, 'rb')
    model = pickle.load(model_dump)
    model_dump.close()
    params, y_size = get_params_by_category(args.category)
    conv_list = args.conv_list
    if not conv_list:
        conv_list = gen_train_data(args.songs_list, args.audio_root, args.gt_root, params, args.conv_root,
                                   args.subsong_len, args.song_len)
    dataloader = get_test_rf_dataloader(conv_list)
    val_rf(model, dataloader, print_results=True)
