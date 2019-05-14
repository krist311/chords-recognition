import _pickle as pickle

from preprocess.params import root_params, maj_min_params, seventh_bass_params, \
    mirex_maj_min_params, mirex_maj_min_bass_params, bass_params, mirex_seventh_params, maj_min_seventh_params, \
    mirex_seventh_bass_params


def get_params_by_category(category):
    if category == 'MirexRoot':
        params = root_params
    elif category == 'MirexMajMin':
        params = mirex_maj_min_params
    elif category == 'maj_min':
        params = maj_min_params
    elif category == 'MirexMajMinBass':
        params = mirex_maj_min_bass_params
    elif category == 'bass':
        params = bass_params
    elif category == 'MirexSevenths':
        params = mirex_seventh_params
    elif category == 'maj_min_7':
        params = maj_min_seventh_params
    elif category == 'MirexSeventhsBass':
        params = mirex_seventh_bass_params
    elif category == 'bass7':
        params = seventh_bass_params
    _, _, _, _, _, y_size, y_ind = params()
    return params, y_size, y_ind


def save_model(model, name):
    output = open(f'pretrained/{name}.pkl', 'wb')
    pickle.dump(model, output, -1)
    output.close()


def load_model(name):
    model_dump = open(name, 'rb')
    model = pickle.load(model_dump)
    model_dump.close()
    return model
