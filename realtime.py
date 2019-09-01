import sys
from time import sleep

import torch

from models import LSTMClassifier
from preprocess.chords import ind_to_chord_names
from utils.parser import get_realtime_parser
from utils.utils import get_params_by_category
import pyaudio
import librosa
import numpy as np

# demo uses best pretrained models for each category
def get_weights_path_by_category(category):
    if category == 'MirexRoot':
        return 'data/predicted/MirexRoot/mod/LSTM_bi_True_MirexRoot_librosa_acc_83.47220891996297_lr_1.0_wd_1e-07_nl_3_hd_128_ne_100_sss_10_sg_0.9_opt_SGD'
    elif category == 'MirexMajMin':
        return 'data/predicted/MirexMajMin/mod/LSTM_bi_True_MirexMajMin_librosa_acc_82.67730773742034_lr_1.0_wd_1e-07_nl_3_hd_128_ne_100_sss_10_sg_0.9_opt_SGD'
    elif category == 'MirexMajMinBass':
        return 'data/predicted/MirexMajMinBass/mod/LSTM_bi_True_MirexMajMinBass_librosa_acc_81.30449231731635_lr_1.0_wd_1e-07_nl_3_hd_128_ne_100_sss_10_sg_0.9_opt_SGD'
    elif category == 'MirexSevenths':
        return 'data/predicted/MirexSevenths/mod/LSTM_bi_True_MirexSevenths_librosa_acc_75.69202799390077_lr_1.0_wd_1e-07_nl_3_hd_128_ne_100_sss_10_sg_0.9_opt_SGD'
    elif category == 'MirexSeventhsBass':
        return 'data/predicted/MirexSeventhsBass/mod/LSTM_bi_True_MirexSeventhsBass_librosa_acc_74.5640614614693_lr_1.0_wd_1e-07_nl_3_hd_128_ne_100_sss_10_sg_0.9_opt_SGD'


def callback(in_data, frame_count, time_info, flag):
    audio_data = np.fromstring(in_data, dtype=np.int16)
    audio_data = librosa.resample(audio_data.astype('float32'), 44100, 11025)
    tuning = librosa.estimate_tuning(y=audio_data, sr=11025)
    X = np.abs(librosa.core.cqt(audio_data, sr=11025, n_bins=84, bins_per_octave=12, tuning=tuning,
                                window='hamming', norm=2)).T
    with torch.no_grad():
        global prev_chord
        if torch.cuda.is_available():
            X = torch.tensor(X).cuda()
        else:
            X = torch.tensor(X)
        X = X.unsqueeze(0)
        pred = model(X)
        y = pred.topk(1, dim=2)[1].squeeze().view(-1)
        from collections import Counter
        counter = Counter(ind_to_chord_names(y, category))
        current_chord = counter.most_common(1)[0][0]
        if prev_chord != current_chord:
            print(current_chord)
            prev_chord = current_chord
    return in_data, pyaudio.paContinue


def predict_stream():
    audio = pyaudio.PyAudio()
    stream = audio.open(format=pyaudio.paInt16,
                        channels=1,
                        rate=44100,
                        input=True,
                        frames_per_buffer=32768,
                        stream_callback=callback)

    stream.start_stream()
    while stream.is_active():
        sleep(0.25)
    stream.close()
    audio.terminate()


if __name__ == '__main__':
    parser = get_realtime_parser()
    args = parser.parse_args(sys.argv[1:])
    category = args.category
    weights_path = get_weights_path_by_category(args.category)
    params, y_size, y_ind = get_params_by_category(args.category)
    prev_chord = ''
    # all best pretrained models have identical architecture
    model = LSTMClassifier(input_size=84, hidden_dim=128, output_size=y_size,
                           num_layers=3,
                           use_gpu=True, bidirectional=True, dropout=[0.4, 0.0, 0.0])
    if torch.cuda.is_available():
        model = model.cuda()
        model.load_state_dict(torch.load(weights_path))
    else:
        model.load_state_dict(torch.load(weights_path, map_location='cpu'))
    model.eval()
    predict_stream()
