# Generate training data based on the ground truth files
# this process leverages the frontend and the ground truth data
import os

from preprocess.params import root_params
import numpy as np

from preprocess.chords import convert_gt, chord_nums_to_inds
from preprocess.frontend import preprocess_audio


def parse_songs_list(songs_list, audio_root, gt_root=None):
    with open(songs_list, 'r') as f:
        songs = []
        for song_folder in f:
            audio_path = audio_root + song_folder
            song_title = song_folder.split('.')[0]
            if gt_root:
                gt_path = gt_root + '/' + song_title + '.lab'
                songs.append((audio_path, gt_path))
            else:
                songs.append((song_title,audio_path))
        return songs


def gen_test_data(songs_list, audio_root, params):
    songs = []
    for song_name, audio_path in parse_songs_list(songs_list, audio_root):
        songs.append((song_name,preprocess_audio(audio_path, params)))
    return songs


def gen_train_data(songs_list, audio_root, gt_root, params, save_as=None):
    param, _, _, _, category = params()
    X, y = [], []
    for audio_path, gt_path in parse_songs_list(songs_list, audio_root, gt_root):
        print('collecting training data of ', audio_path)
        X_song = preprocess_audio(audio_path, param)
        # ** ** ** map audio content to gt ** ** **
        y_nums, inds_to_remove = convert_gt(gt_path, param['hop_size'], param['fs'], len(X_song), category)
        np.delete(X_song, np.r_[inds_to_remove])
        # TODO transpose chords
        y_song = chord_nums_to_inds(y_nums, category)
        X.extend(X_song)
        y.extend(y_song)
    if save_as:
        np.savetxt(save_as, np.append(X, np.array([y]).T, axis=1),
                   delimiter=",", fmt='%s')
    return X, y
