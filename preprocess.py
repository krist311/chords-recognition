import sys

from preprocess.generators import gen_train_data
from utils.parser import get_preprocess_parser
from utils.utils import get_params_by_category
import glog as log
from pprint import pformat


if __name__ == '__main__':
    parser = get_preprocess_parser()
    args = parser.parse_args(sys.argv[1:])
    log.info('Arguments:\n' + pformat(args.__dict__))
    params, _ = get_params_by_category(args.category)
    conv_root = args.conv_root
    if args.use_librosa:
        conv_root = conv_root + '/librosa/'
    else:
        conv_root = conv_root + '/mauch/'
    conv_list = gen_train_data(args.songs_list, args.audio_root, args.gt_root, params, conv_root,
                               args.subsong_len, args.song_len, use_librosa=args.use_librosa)
    print(conv_list)
