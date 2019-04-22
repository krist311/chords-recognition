import argparse
import sys

from preprocess.generators import gen_train_data
from preprocess.params import root_params, maj_min_params, maj_min_bass_params, seventh_params, seventh_bass_params


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
    return params


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--songs_list', type=str, required=True)
    parser.add_argument('--audio_root', default='data/audio/', type=str)
    parser.add_argument('--gt_root', default='data/gt/', type=str)
    parser.add_argument('--conv_root', default='data/converted/', type=str)
    parser.add_argument('--category', default='MirexRoot', type=str)
    parser.add_argument('--subsong_len', default=40, type=int)
    parser.add_argument('--song_len', default=180, type=int)
    parser.add_argument('--use_librosa', default=True, type=bool)
    return parser


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args(sys.argv[1:])
    params = get_params_by_category(args.category)
    conv_root = args.conv_root
    if args.use_librosa:
        conv_root = conv_root + '/librosa/'
    else:
        conv_root = conv_root + '/mauch/'
    conv_list = gen_train_data(args.songs_list, args.audio_root, args.gt_root, params, conv_root,
                               args.subsong_len, args.song_len, use_librosa=args.use_librosa)
    print(conv_list)
