"""--tchord--            --digit--     --dif--     --weight--    --stype--
 ************************* uni ******************************** 
 1->1 1               1                0            0          0
 ************************* dyad ******************************* 
 1->b2 b2             1,2              1            1          0
 1->2 2               1,3              2            1          0
 1->b3 b3             1,4              3            1          2
 1->3 3               1,5              4            1          1
 1->4 4               1,6              5            1          0
 1->4# 4#             1,7              6            1          0
 1->5 5               1,8              7            1          0
 1->b6 b6             1,9              8            1          0
 1->6 6               1,10             9            1          0
 1->b7 b7             1,11             10           1          0
 1->7 7+              1,12             11           1          0
 ************************* major and minor ************************ 
 1->3->5 maj          1,5,8            4,7          1,1        1
 1->b3->5 min         1,4,8            3,7          1,1        2
 ************************ suspend and add *************************** 
 1->2->5 sus2         1,3,8            2,7          1,1        0
 1->4->5 sus4         1,6,8            5,7          1,1        0
 1->2->3->5 add9      1,3,5,8          2,4,7        [1-tc]*3   1
 1->3->4->5 add11     1,5,6,8          4,5,7        [1-tc]*3   1
 1->2->b3->5 madd9    1,3,4,8          2,3,7        [1-tc]*3   2
 1->b3->4->5 madd11   1,4,6,8          3,5,7        [1-tc]*3   2
 1->2->5->7 maj7sus2  1,3,8,12         2,7,11       [1-tc]*3   0
 1->2->5->b7 7sus2    1,3,8,11         2,7,10       [1-tc]*3   0
 1->4->5->7 maj7sus4  1,6,8,12         5,7,11       [1-tc]*3   0
 1->4->5->b7 7sus4    1,6,8,11         5,7,10       [1-tc]*3   0
 1->4->5->7->9 maj9sus4 1,6,8,12,15    5,7,11,14    [1-pc]*4   0
 1->4->5->b7->9 9sus4 1,6,8,11,15      5,7,10,14    [1-pc]*4   0
 ************************ sixth ******************************* 
 1->3->5->6 maj6      1,5,8,10         4,7,9        [1-tc]*3   1
 1->b3->5->6 min6     1,4,8,10         3,7,9        [1-tc]*3   2
 ************************ sevenths ***************************** 
 1->3->5->7 maj7      1,5,8,12         4,7,11       [1-tc]*3   1
 1->b3->5->b7 min7    1,4,8,11         3,7,10       [1-tc]*3   2
 1->3->5->b7 7        1,5,8,11         4,7,10       [1-tc]*3   1
 ************************ Extended ******************************
 1->3->5->7->9 maj9   1,5,8,12,15      4,7,11,14    [1-pc]*4   1
 1->b3->5->b7->9 min9 1,4,8,11,15      3,7,10,14    [1-pc]*4   2
 1->3->5->b7->9 9     1,5,8,11,15      4,7,10,14    [1-pc]*4   1
 1->3->5->7->9->11 maj11   1,5,8,12,15,18      4,7,11,14,17    [1-hc]*5   1
 1->b3->5->b7->9->11 min11 1,4,8,11,15,18      3,7,10,14,17    [1-hc]*5   2
 1->3->5->b7->9->11 11     1,5,8,11,15,18      4,7,10,14,17    [1-hc]*5   1
 ******************** augmented and diminished ******************** 
 1->3->5# aug         1,5,9            4,8          1,1        1
 1->b3->b5 dim        1,4,7            3,6          1,1        2
 1->b3->b5->bb7 dim7  1,4,7,10         3,6,9        [1-tc]*3   2
 1->b3->b5->b7 hdim   1,4,7,11         3,6,10       [1-tc]*3   2
 1->b3->5->7 minmaj7  1,4,8,12         3,7,11       [1-tc]*3   2
 ************************ majminbass *************************** 
 1->b3->b6 maj/3      1,4,9            3,8          1,1        0
 1->4->6 maj/5        1,6,10           5,9          1,1        0
 1->3->6 min/b3       1,5,10           4,9          1,1        0
 1->4->b6 min/5       1,6,9            5,8          1,1        0
 ************************ seventhbass *************************** 
 1->b3->5->b6 maj7/3  1,4,8,9          3,7,8        [1-tc]*3   0
 1->3->4->6 maj7/5    1,5,6,10         4,5,9        [1-tc]*3   0
 1->b2->4->b6 maj7/7  1,2,6,9          1,5,8        [1-tc]*3   0
 1->3->5->6 min7/b3   1,5,8,10         4,7,9        [1-tc]*3   0
 1->b3->4->b6 min7/5  1,4,6,9          3,5,8        [1-tc]*3   0
 1->2->4->6 min7/b7   1,3,6,10         2,5,9        [1-tc]*3   0
 1->b3->b5->b6 7/3    1,4,7,9          3,6,8        [1-tc]*3   0
 1->b3->4->6 7/5      1,4,6,10         3,5,9        [1-tc]*3   0
 1->2->b5->6 7/b7     1,3,7,10         2,6,9        [1-tc]*3   0
 ************************ otherslash *************************** 
 1->2->4->b7 maj/2    1,3,6,11         2,5,10       [1-tc]*3   0
 1->b2->4->b7 min/2   1,2,6,11         1,5,10       [1-tc]*3   0

 if this file is modified, also check file "trebleTypeMapping.m" and
 "cast2MajMin.m"

 for naming conventions and labeling methodology, please refer to:
 Harte, C., M. Sandler, S. Abdallah, and E. Gуmez. 2005.
 “Symbolic representation of musical chords: A proposed syntax for text
 annotations.” In Proceedings of the 6th International Society for Music
 Information Retrieval Conference (ISMIR), 66–71."""
import os

import numpy as np


def convert_gt(gt_path, hop_size, fs, song_len, category):
    tw = hop_size / fs  # time ticks
    y, inds_to_remove = [], []
    with open(gt_path, 'r') as f:
        for gt_chord_line in f:
            # decipher start time, end time and chord
            gt_chord_line = gt_chord_line.rstrip()
            str_toks = gt_chord_line.split(' ')
            if len(str_toks) == 1:
                str_toks = gt_chord_line.split('\t')
            start_time = float(str_toks[0])
            end_time = float(str_toks[1])
            chord = str_toks[2]
            sb = round(start_time / tw)
            eb = round(end_time / tw)
            # cast chord to standard chords

            ch_num = chord_to_categories(chord)
            y.extend((eb - sb) * [ch_num])
        if len(y) < song_len:
            y.extend((song_len - len(y)) * [[-1] * 9])
        else:
            y = y[:song_len]
    return y


def chord_to_categories(chord):
    # categories = ('MirexRoot', 'MirexMajMin', 'MirexMajMinBass', 'MirexSevenths', 'MirexSeventhsBass')
    # add root category
    nums = [chord_to_nums(chord, 'MirexRoot')]

    # add maj_min category and type of maj|min chord - [-1 - cannot be converted, 0 - no chord, 1 - maj, 2 - min]
    maj_min, chord_type, bass = chord_to_nums(chord, 'MirexMajMin')
    nums.append(maj_min)
    chord_type = TypesConverter.maj_min_or_maj_min_seven_type_to_ind(chord_type)
    nums.append(chord_type)

    # add maj_min bass category and bass presence [-1 -cannot be converted, 0 -no chord, 1 - no bass, 2 - 3, 3 - 5)
    maj_min_bass, chord_type, bass = chord_to_nums(chord, 'MirexMajMinBass')
    nums.append(maj_min_bass)
    bass = TypesConverter.bass_to_ind(bass)
    nums.append(bass)

    # add maj_min_seventh category
    maj_min_seventh, chord_type, bass = chord_to_nums(chord, 'MirexSevenths')
    nums.append(maj_min_seventh)
    chord_type = TypesConverter.maj_min_or_maj_min_seven_type_to_ind(chord_type)
    nums.append(chord_type)

    # add maj_min_seventh_bass category
    maj_min_seventh, chord_type, bass = chord_to_nums(chord, 'MirexSeventhsBass')
    nums.append(maj_min_seventh)
    bass = TypesConverter.bass_to_ind(bass)
    nums.append(bass)

    return nums


def ind_to_chord_names(inds, category):
    _, ind_to_name = create_chords_list(category)
    return [ind_to_name[ind] for ind in inds]


def preds_to_lab(y, hop_size, fs, category, save_path, song_name):
    results = []
    start_time = 0.0
    chord_names = ind_to_chord_names(y, category)
    tw = (hop_size / fs)  # time ticks
    y_prev = chord_names[0]
    for i, chord_name in enumerate(chord_names, 1):
        if chord_name == y_prev and i != len(chord_names):
            continue
        end_time = i * tw
        results.append(f"{start_time}	{end_time}	{y_prev}")
        start_time = end_time
        y_prev = chord_name
    if save_path:
        predicted_path = f"{save_path}/{song_name}.lab"
        # create folder for saving predictions
        os.makedirs(predicted_path[:-len(predicted_path.split('/')[-1])], exist_ok=True)
        np.savetxt(predicted_path, results, delimiter=",", fmt='%s')
    return results


def chords_nums_to_inds(chords_nums):
    # chords_nums - [root, MirexMajMin, maj/min, MirexMajMinBass, 3/5 bass, MirexSevenths, maj/min/7, MirexSeventhsBass, 3/5/7 bass]
    chords_nums = np.array(chords_nums, dtype=str)
    chords_nums[:, 1] = chord_nums_to_inds(chords_nums[:, 1], 'MirexMajMin')
    chords_nums[:, 3] = chord_nums_to_inds(chords_nums[:, 3], 'MirexMajMinBass')
    chords_nums[:, 5] = chord_nums_to_inds(chords_nums[:, 5], 'MirexSevenths')
    chords_nums[:, 7] = chord_nums_to_inds(chords_nums[:, 7], 'MirexSeventhsBass')
    return chords_nums


def chord_nums_to_inds(chord_nums, category):
    num_to_ind, _ = create_chords_list(category)
    return [num_to_ind[chord_num] for chord_num in chord_nums]


def create_chords_list(category):
    chord_types = get_chord_types_by_category(category)
    num_to_ind = {'-1': -1, '0': 0}
    ind_to_name = ['N']

    for note in get_note_names():
        if category == 'MirexRoot':
            num_to_ind[note2num(note)] = len(ind_to_name)
            ind_to_name.append(note)
        else:
            for chord_type in chord_types:
                for bass in get_inversions(category, chord_type[0]):
                    num_to_ind[f"{note2num(note)}:{chord_type[0]}{bass}"] = len(ind_to_name)
                    ind_to_name.append(f"{note}:{chord_type[0]}{bass}")

    return num_to_ind, ind_to_name


def get_inversions(category, chord_type):
    if category == 'MirexMajMinBass' or category == 'MirexSeventhsBass':
        if 'maj' == chord_type:
            return '', '/3', '/5'
        if 'min' == chord_type:
            return '', '/b3', '/5'
        if 'maj7' == chord_type:
            return '', '/3', '/5', '/7'
        if 'min7' == chord_type:
            return '', '/b3', '/5', '/b7'
        if '7' == chord_type:
            return '', '/3', '/5', '/b7'
    else:
        return '',


def get_chord_params_by_mirex_category(category):
    return {
        'MirexRoot': {
            'controls': {
                'triadcontrol': 0,
                'tetradcontrol': 0.25,
                'pentacontrol': 0,
                'hexacontrol': 0},
            'types_and_inv': {
            }
        },
        'MirexMajMin': {
            'controls': {
                'triadcontrol': 0,
                'tetradcontrol': 0.25,
                'pentacontrol': 0,
                'hexacontrol': 0},
            'types_and_inv': {
                # is inversion for type supported
                'maj': False,
                'min': False
            }
        },
        'MirexMajMinBass': {
            'controls': {
                'triadcontrol': 0,
                'tetradcontrol': 0.25,
                'pentacontrol': 0,
                'hexacontrol': 0},
            'types_and_inv': {
                # is inversion for type supported
                'maj': True,
                'min': True
            }
        },
        'MirexSevenths': {
            'controls': {
                'triadcontrol': 0,
                'tetradcontrol': 0.25,
                'pentacontrol': 0,
                'hexacontrol': 0},
            'types_and_inv': {
                # is inversion for type supported
                'maj': False,
                'min': False,
                '7': False,
                'maj7': False,
                'min7': False
            }
        },
        'MirexSeventhsBass': {
            'controls': {
                'triadcontrol': 0,
                'tetradcontrol': 0.25,
                'pentacontrol': 0,
                'hexacontrol': 0},
            'types_and_inv': {
                # is inversion for type supported
                'maj': True,
                'min': True,
                '7': True,
                'maj7': True,
                'min7': True
            }
        },
    }.get(category)


def split_chord(chord):
    root, type_name, adds, bass = chord, '', [], ''
    if '/' in chord:
        chord, bass = chord.split('/')
        root = chord
    if '(' in chord:
        adds = chord[chord.find('(') + 1:chord.find(')')].split(',')
        chord = chord[:chord.find('(')] + chord[chord.find(')') + 1:]
        root = chord
    if ':' in chord:
        root, type_name = chord.split(':')
    if not type_name and not adds:
        type_name = 'maj'

    return root, type_name, adds, bass


def get_types_and_inv(category):
    return get_chord_params_by_mirex_category(category)['types_and_inv']


def chord_to_nums(chord, category):
    root, type_name, adds, bass = split_chord(chord)
    root = note2num(root)
    if category == 'MirexRoot':
        return root
    if root == 0:
        return 0, 0, 0
    if len(adds) > 0 or type_name not in get_types_and_inv(category):
        comps = list(get_components_by_notation(type_name))
        for add in adds:
            if '*' in add:
                # remove this addition
                ind = comps.index(add[1:])
                del comps[ind]
            else:
                # add this addition
                comps.append(add)
        type_name = comps_to_type(comps, category)
    if not type_name:
        return -1, -1, -1
    if not get_types_and_inv(category)[type_name] \
            or (category == 'MirexMajMinBass' and ('3' not in bass and '5' != bass)) \
            or (category == 'MirexSeventhsBass' and ('3' not in bass and '5' != bass and '7' not in bass)):
        bass = ''
    if '7' in bass and '7' not in type_name:
        bass = ''
    return f"{root}:{type_name}{f'/{bass}' if bass else ''}", type_name, bass


def comps_to_type(comps, category):
    supported_types = get_chord_types_by_category(category)
    for type_name, sup_comps in sorted(supported_types, key=lambda x: len(supported_types[1]), reverse=True):
        is_subset = True
        for sup_comp in sup_comps:
            if sup_comp not in comps:
                is_subset = False
                break
        if is_subset:
            return type_name
    return False


def get_components_by_notation(type_name):
    return {
        "": (),
        # ************************* major and minor ************************
        'maj': ('1', '3', '5'),
        'min': ('1', 'b3', '5'),
        # ************************ sevenths *****************************
        '7': ('1', '3', '5', 'b7'),
        'maj7': ('1', '3', '5', '7'),
        'min7': ('1', 'b3', '5', 'b7'),
        # ************************ suspend and add ***************************
        'sus2': ('1', '2', '5'),
        'sus4': ('1', '4', '5'),
        'add9': ('1', '2', '3', '5'),
        'add11': ('1', '3', '4', '5'),
        'madd9': ('1', '2', 'b3', '5'),
        'madd11': ('1', 'b3', '4', '5'),
        'maj7sus2': ('1', '2', '5', '7'),
        '7sus2': ('1', '2', '5', 'b7'),
        'maj7sus4': ('1', '4', '5', '7'),
        '7sus4': ('1', '4', '5', 'b7'),
        'maj9sus4': ('1', '4', '5', '7', '9'),
        '9sus4': ('1', '4', '5', 'b7', '9'),
        # ************************ sixth ******************************* #
        'maj6': ('1', '3', '5', '6'),
        'min6': ('1', 'b3', '5', '6'),
        # ************************ Extended ******************************#
        'maj9': ('1', '3', '5', '7', '9'),
        'min9': ('1', 'b3', '5', 'b7', '9'),
        '9': ('1', '3', '5', 'b7', '9'),
        'maj11': ('1', '3', '5', '7', '9', '11'),
        'min11': ('1', 'b3', '5', 'b7', '9', '11'),
        '11': ('1', '3', '5', 'b7', '9', '11'),
        'maj13': ('1', '3', '5', '7', '9', '11', '13'),
        '13': ('1', '3', '5', 'b7', '9', '11', '13'),
        # ******************** augmented and diminished ******************** #
        'aug': ('1', '3', '5#'),
        'dim': ('1', 'b3', 'b5'),
        'dim7': ('1', 'b3', 'b5', 'bb7'),
        'hdim': ('1', 'b3', 'b5', 'b7'),
        # HACK can't find its structure, but it couldnt be converted to any MIREX category
        'hdim7': ('10', '20', '30', '40'),
        'minmaj7': ('1', 'b3', '5', '7')
    }.get(type_name)


def get_chord_types_by_category(category):
    types = []
    for type_name, inv in get_chord_params_by_mirex_category(category)['types_and_inv'].items():
        types.append((type_name, get_components_by_notation(type_name)))
    return types


def get_note_names():
    return 'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'


def note2num(note):
    return {'C': '1', 'C#': '2', 'Db': '2', 'D': '3', 'D#': '4', 'Eb': '4',
            'E': '5', 'Fb': '5', 'F': '6', 'F#': '7', 'Gb': '7', 'G': '8', 'G#': '9', 'Ab': '9', 'A': '10', 'A#': '11',
            'Bb': '11', 'B': '12',
            'Cb': '12'}.get(note, 0)


class TypesConverter:
    @staticmethod
    def maj_min_or_maj_min_seven_type_to_ind(chord_type):
        if type(chord_type) == str:
            if chord_type == 'maj':
                return 1
            if chord_type == 'min':
                return 2
            if chord_type == '7':
                return 3
            if chord_type == 'maj7':
                return 4
            if chord_type == 'min7':
                return 5
        else:
            return chord_type

    @staticmethod
    def bass_to_ind(bass):
        if type(bass) == str:
            if bass == '':
                return 1
            if bass == '3' or bass == 'b3':
                return 2
            if bass == '5':
                return 3
            if bass == '7' or bass == 'b7':
                return 4
        else:
            return bass

    @staticmethod
    def ind_to_bass(ind):
        return {1: '', 2: '3', 3: '5', 4: '7'}.get(ind)

    @staticmethod
    def ind_to_type(ind):
        return {1: 'maj', 2: 'min', 3: '7', 4: 'maj7', 5: 'min7'}.get(ind)
