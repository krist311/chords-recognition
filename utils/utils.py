import _pickle as pickle

from preprocess.params import root_params, maj_min_params, maj_min_bass_params, seventh_params, seventh_bass_params


def get_params_by_category(category):
    params, y_size = 0, 0
    if category == 'MirexRoot':
        params = root_params
    elif category == 'MirexMajMin':
        params = maj_min_params
    elif category == 'MirexMajMinBass':
        params = maj_min_bass_params
    elif category == 'MirexSevenths':
        params = seventh_params
    elif category == 'MirexSeventhsBass':
        params = seventh_bass_params
    _, _, _, _, _, y_size = params()
    return params, y_size


def save_model(model, name):
    output = open(f'pretrained/{name}.pkl', 'wb')
    pickle.dump(model, output, -1)
    output.close()


def load_model(name):
    model_dump = open(name, 'rb')
    model = pickle.load(model_dump)
    model_dump.close()
    return model
