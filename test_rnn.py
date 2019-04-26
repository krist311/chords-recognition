import sys

from dataloader import get_test_seq_dataloader
from preprocess.generators import gen_train_data
from train_rnn import val_model
from utils.parser import get_test_parser
from utils.utils import get_params_by_category, load_model

if __name__ == '__main__':
    parser = get_test_parser()
    args = parser.parse_args(sys.argv[1:])
    model = load_model(args.model)
    params, y_size = get_params_by_category(args.category)
    conv_list = args.conv_list
    if not conv_list:
        conv_list = gen_train_data(args.songs_list, args.audio_root, args.gt_root, params, args.conv_root,
                                   args.subsong_len, args.song_len)
    dataloader = get_test_seq_dataloader(conv_list)
    val_model(model, dataloader, print_results=True, num_classes=y_size)
