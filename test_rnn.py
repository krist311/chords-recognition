import sys

import torch

from dataloader import get_test_seq_dataloader
from models import LSTMClassifier
from preprocess.chords import preds_to_lab
from preprocess.generators import gen_train_data, gen_test_data
from train_rnn import val_model
from utils.parser import get_test_parser
from utils.utils import get_params_by_category, load_model


def t(model, songs_list, audio_root, params, save_path):
    param, _, _, _, category, _, _ = params()
    for song_name, X in gen_test_data(songs_list, audio_root, param):
        with torch.no_grad():
            pred = model(torch.tensor(X).cuda())
            y = pred.topk(1, dim=2)[1].squeeze().view(-1)
            preds_to_lab(y, param['hop_size'], param['fs'], category, save_path, song_name)


if __name__ == '__main__':
    parser = get_test_parser()
    args = parser.parse_args(sys.argv[1:])
    model = torch.load(args.model)
    model.eval()

    params, y_size, y_ind = get_params_by_category(args.category)
    model = LSTMClassifier(input_size=84, hidden_dim=128, output_size=y_size,
                           num_layers=args.num_layers,
                           use_gpu=True, bidirectional=True, dropout=[0.4,0.0,0.0])

    model.load_state_dict(torch.load(args.model))
    model.eval()
    conv_list = args.conv_list
    if args.save_as_lab:
        t(model, args.test_list, args.audio_root, params, args.lab_path)
    else:
        if not conv_list:
            conv_list = gen_train_data(args.test_list, args.audio_root, args.gt_root, params, args.conv_root,
                                       args.subsong_len, args.song_len)
        dataloader = get_test_seq_dataloader(conv_list)
        val_model(model, dataloader, print_results=True, num_classes=y_size)
