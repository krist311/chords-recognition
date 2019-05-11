import argparse


def get_base_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio_root', default='data/audio/', type=str)
    parser.add_argument('--gt_root', default='data/gt/', type=str)
    parser.add_argument('--conv_root', default='data/converted/', type=str,
                        help='determines folder where converted datasets will be saved')
    parser.add_argument('--category', default='MirexRoot', type=str)
    parser.add_argument('--subsong_len', default=40, type=int, )
    parser.add_argument('--song_len', default=180, type=int,
                        help='if --subsong_len is not specified, song will be cutted or zeropaded to <i>song_len</i>')
    parser.add_argument('--use_librosa', default=True, type=bool,
                        help='by default librosa.cqt will be used for preprocessing')
    return parser


def get_preprocess_parser():
    parser = get_base_parser()
    parser.add_argument('--songs_list', type=str, required=True)
    parser.add_argument('-num_bins', type=int, default=84)
    return parser


def get_train_parser():
    parser = get_base_parser()
    parser.add_argument('--songs_list', default='data/tracklists/FullList386.txt', type=str)
    parser.add_argument('--conv_list', type=str)
    parser.add_argument('--save_model', type=bool, default=False,
                        help='If true, model will be saved in pretrained folder')
    return parser


def get_train_rf_parser():
    parser = get_train_parser()
    parser.add_argument('--criterion', default='entropy', type=str)
    parser.add_argument('--max_features', default='log2', type=str)
    parser.add_argument('--n_estimators', default='1', type=int)
    return parser


def get_train_rnn_parser():
    parser = get_train_parser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--opt', type=str, default='Adam', help='optimizer:SGD or Adam, default: Adam')
    parser.add_argument('--num_epochs', default=2, type=int)
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--weight_decay', default=1e-5, type=float)
    parser.add_argument('--hidden_dim', default=50, type=int)
    parser.add_argument('--num_layers', default=2, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--bidirectional', default=False, type=bool)
    parser.add_argument('--sch_step_size', default=100, type=int)
    parser.add_argument('--sch_gamma', default=0.1, type=float)
    parser.add_argument('--val_step', default=10, type=int)
    parser.add_argument('--momentum', default=0.8, type=float, help='SGD momentum')

    return parser


def get_test_parser():
    parser = get_base_parser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--conv_list', type=str, required=True)
    return parser
