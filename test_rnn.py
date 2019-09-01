import sys

import torch

from models import LSTMClassifier
from preprocess.chords import preds_to_lab
from preprocess.generators import gen_test_data
from utils.parser import get_test_parser
from utils.utils import get_params_by_category


def t(model, songs_list, audio_root, params, save_path):
    param, _, _, _, category, _, _ = params()
    for song_name, X in gen_test_data(songs_list, audio_root, param):
        with torch.no_grad():
            if torch.cuda.is_available():
                X = torch.tensor(X).cuda()
            else:
                X = torch.tensor(X)
            pred = model(X)
            y = pred.topk(1, dim=2)[1].squeeze().view(-1)
            preds_to_lab(y, param['hop_size'], param['fs'], category, save_path, song_name)


if __name__ == '__main__':
    parser = get_test_parser()
    args = parser.parse_args(sys.argv[1:])

    params, y_size, y_ind = get_params_by_category(args.category)
    model = LSTMClassifier(input_size=args.input_size, hidden_dim=args.hidden_dim, output_size=y_size,
                           num_layers=args.num_layers,
                           use_gpu=torch.cuda.is_available(), bidirectional=args.is_bidirectional, dropout=args.dropout)
    if torch.cuda.is_available():
        model = model.cuda()
        model.load_state_dict(torch.load(args.model))
    else:
        model.load_state_dict(torch.load(args.model, map_location='cpu'))
    model.eval()
    conv_list = args.conv_list
    t(model, args.test_list, args.audio_root, params, args.lab_path)
