# Generate training data based on the ground truth files
# this process leverages the frontend and the ground truth data
import os

import math
import numpy as np

from preprocess.chords import convert_gt, chord_nums_to_inds, chords_nums_to_inds
from preprocess.frontend import preprocess_mauch, preprocess_librosa


def iter_songs_list(songs_list):
    with open(songs_list, 'r') as f:
        for song_folder in f:
            song_folder = song_folder.rstrip()
            song_title = song_folder[:-len(song_folder.split('.')[-1]) - 1]
            yield song_title, song_folder


def gen_test_data(songs_list, audio_root, params):
    for song_name, audio_path in iter_songs_list(songs_list):
        yield (song_name, preprocess_librosa(f'{audio_root}/{audio_path}', params, mod_steps=(0,)))


def gen_train_data_old(songs_list, audio_root, gt_root, params, converted_root=None, subsong_len=None, song_len=180,
                       use_librosa=False):
    def split(data, subsong_len):
        # convert len in seconds to indexes
        inds_len = int(subsong_len * (param['fs'] / param['hop_size']))
        return np.array_split(data, round(len(data) / inds_len))

    param, _, _, _, category, _ = params()
    converted_list = []
    X, y = [], []
    for song_title, audio_path in iter_songs_list(songs_list):
        print('collecting training data of ', song_title)
        X_song = preprocess_mauch(f'{audio_root}/{audio_path}', param, use_librosa).T
        # ** ** ** map audio content to gt ** ** **
        y_nums, inds_to_remove = convert_gt(f'{gt_root}/{song_title}.lab', param['hop_size'], param['fs'], len(X_song),
                                            category)
        # remove chords which couldn't be converted to current category
        X_song = np.delete(X_song, np.r_[inds_to_remove], axis=0)
        # TODO transpose chords
        y_song = chord_nums_to_inds(y_nums, category)

        data = np.append(X_song, np.array([y_song]).T, axis=1)
        converted_path = f'{converted_root}/{song_title}'
        os.makedirs(converted_path[:-len(converted_path.split('/')[-1])], exist_ok=True)
        if subsong_len:
            for i, data_part in enumerate(split(data, subsong_len)):
                converted_list.append(f"{converted_path}_part{i}.csv")
                np.savetxt(f"{converted_path}_part{i}.csv", data_part,
                           delimiter=",", fmt='%s')
        else:
            inds_len = int(song_len * (param['fs'] / param['hop_size']))
            data = np.pad(data, ((0, song_len - len(data)), (0, 0)), 'constant') if len(data) < inds_len else data[
                                                                                                              :inds_len]
            np.savetxt(converted_path + '.csv', data, delimiter=",", fmt='%s')
            converted_list.append(converted_path + '.csv')

        # save list of converted songs
    if converted_list:
        conv_alg_name = 'librosa'
        if not use_librosa:
            conv_alg_name = 'mauch'
        converted_list_name = f"{songs_list.split('/')[-1].split('.')[0]}_converted_{conv_alg_name}.txt"
        np.savetxt(converted_list_name, converted_list, fmt='%s')
        return converted_list_name
    else:
        return X, y


def gen_train_data(songs_list, audio_root, gt_root, params, converted_root=None, subsong_len=None, song_len=180,
                   mod_steps=(0,), use_librosa=True):
    def split(data, subsong_len):
        # convert len in seconds to indexes
        inds_len = int(subsong_len * (param['fs'] / param['hop_size']))
        return np.array_split(data, math.ceil(len(data) / inds_len))

    def mod_y(y, mod_step):
        y_mod = []
        mod_step = mod_step % 12
        for chords in y:
            mod_row=[]
            for chord in chords:
                if type(chord) == str:
                    if ':' in chord:
                        root, chord_type = chord.split(':')
                    else:
                        root, chord_type = chord, ''
                    root = int(root)
                    if root + mod_step < 1:
                        root = 12 - (root + mod_step)
                    elif root + mod_step > 12:
                        root = root + mod_step - 12
                    else:
                        root = root + mod_step
                    if chord_type:
                        mod_row.append(f'{root}:{chord_type}')
                    else:
                        mod_row.append(root)
                else:
                    mod_row.append(chord)
            y_mod.append(mod_row)
        return y_mod

    param, _, _, _, category, _, _ = params()
    converted_list = []
    for song_title, audio_path in iter_songs_list(songs_list):
        # create folder for converted audio
        converted_path = f'{converted_root}/{song_title}'
        os.makedirs(converted_path[:-len(converted_path.split('/')[-1])], exist_ok=True)

        print('collecting training data of ', song_title)
        Xs = preprocess_librosa(f'{audio_root}/{audio_path}', param, mod_steps=mod_steps)
        # ** ** ** map audio content to gt ** ** **
        y_nums = convert_gt(f'{gt_root}/{song_title}.lab', param['hop_size'], param['fs'],
                            len(Xs[0]),
                            category)
        # TODO transpose chords
        for i, mod_step in enumerate(mod_steps):
            y_nums_song = mod_y(y_nums, mod_step)
            y_song = chords_nums_to_inds(y_nums_song)
            data = np.append(Xs[i], y_song, axis=1)

            if subsong_len:
                for i, data_part in enumerate(split(data, subsong_len)):
                    converted_list.append(f"{converted_path}_mod{mod_step}_part{i}.csv")
                    np.savetxt(f"{converted_path}_mod{mod_step}_part{i}.csv", data_part,
                               delimiter=",", fmt='%s')
        # save list of converted songs
    conv_alg_name = 'librosa'
    converted_list_name = f"{songs_list.split('/')[-1].split('.')[0]}_converted_{conv_alg_name}.txt"
    np.savetxt(converted_list_name, converted_list, fmt='%s')
    return converted_list_name
