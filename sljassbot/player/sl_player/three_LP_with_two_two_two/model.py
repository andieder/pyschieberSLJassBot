import os
import os.path

from keras.models import load_model


# TODO: first 2 Conv1D then 2 Fully
def build_model(model_path, learning_rate=0.01):
    if model_path is None:
        return None
    if os.path.exists(model_path):
        model = load_model(model_path)
        print('Load existing model.')
    else:
        print('There is no model to load.')
        model = None
    return model
