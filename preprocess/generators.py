# Generate training data based on the ground truth files
# this process leverages the frontend and the ground truth data
import os

from preprocess.params import root_params
import numpy as np

from preprocess.chords import convert_gt, chord_nums_to_inds
from preprocess.frontend import preprocess_audio


def iter_songs_list(songs_list, ):
    with open(songs_list, 'r') as f:
        for song_folder in f:
            song_folder = song_folder.rstrip()
            song_title = song_folder[:-len(song_folder.split('.')[-1]) -1]
            yield song_title, song_folder


def gen_test_data(songs_list, audio_root, params):
    songs = []
    for song_name, audio_path in iter_songs_list(songs_list, audio_root):
        songs.append((song_name, preprocess_audio(audio_path, params)))
    return songs


def gen_train_data(songs_list, audio_root, gt_root, params, converted_root=None):
    param, _, _, _, category, _ = params()
    converted_list = []
    for song_title, audio_path in iter_songs_list(songs_list):
        print('collecting training data of ', song_title)
        X = preprocess_audio(f'{audio_root}/{audio_path}', param)
        # ** ** ** map audio content to gt ** ** **
        y_nums, inds_to_remove = convert_gt(f'{gt_root}/{song_title}.lab', param['hop_size'], param['fs'], len(X),
                                            category)
        np.delete(X, np.r_[inds_to_remove])
        # TODO transpose chords
        y = chord_nums_to_inds(y_nums, category)
        if converted_root:
            converted_path = f'{converted_root}/{song_title}.csv'
            os.makedirs(converted_path[:-len(converted_path.split('/')[-1])], exist_ok=True)
            np.savetxt(converted_path, np.append(X, np.array([y]).T, axis=1),
                       delimiter=",", fmt='%s')
            converted_list.append(converted_path)
        # save list of converted songs
        if converted_list:
            converted_name = f"{songs_list.split('/')[-1].split('.')[0]}_converted.txt"
            np.savetxt(converted_name, converted_list, fmt='%s')
