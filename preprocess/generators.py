# Generate training data based on the ground truth files
# this process leverages the frontend and the ground truth data
import os
import numpy as np

from preprocess.chords import convert_gt, chord_nums_to_inds
from preprocess.frontend import preprocess_audio


def iter_songs_list(songs_list):
    with open(songs_list, 'r') as f:
        for song_folder in f:
            song_folder = song_folder.rstrip()
            song_title = song_folder[:-len(song_folder.split('.')[-1]) - 1]
            yield song_title, song_folder


def gen_test_data(songs_list, audio_root, params, use_librosa):
    songs = []
    for song_name, audio_path in iter_songs_list(songs_list):
        songs.append((song_name, preprocess_audio(audio_path, params, use_librosa)))
    return songs


def gen_train_data(songs_list, audio_root, gt_root, params, converted_root=None, subsong_len=None, song_len=180, use_librosa=False):
    def split(data, subsong_len):
        # convert len in seconds to indexes
        inds_len = int(subsong_len * (param['fs'] / param['hop_size']))

        def zero_pad(data, inds_len):
            tail = len(data) % inds_len
            if tail:
                data = np.pad(data, ((0, inds_len - tail), (0, 0)), 'constant') if tail > inds_len / 2 else data[:-tail]
            return data

        data = zero_pad(data, inds_len)
        return np.split(data, len(data) / inds_len)

    param, _, _, _, category, _ = params()
    converted_list = []
    X, y = [], []
    for song_title, audio_path in iter_songs_list(songs_list):
        print('collecting training data of ', song_title)
        X_song = preprocess_audio(f'{audio_root}/{audio_path}', param, use_librosa).T
        # ** ** ** map audio content to gt ** ** **
        y_nums, inds_to_remove = convert_gt(f'{gt_root}/{song_title}.lab', param['hop_size'], param['fs'], len(X_song),
                                            category)
        # remove chords which couldn't be converted to current category
        np.delete(X_song, np.r_[inds_to_remove])
        # TODO transpose chords
        y_song = chord_nums_to_inds(y_nums, category)
        if converted_root:
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
        else:
            X.append(X_song)
            y.append(y_song)
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
