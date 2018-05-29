import os
import os.path

from keras.layers import Dense, Flatten, Conv1D, Reshape
from keras.optimizers import Nadam
from keras.models import Sequential
from keras.models import load_model
from keras.regularizers import l2
from keras import backend as K
from keras.losses import mean_squared_error

from sljassbot.player.rl_player.input_handler import InputHandler


def huber_loss(a, b, in_keras=True):
    error = a - b
    quadratic_term = error * error / 2
    linear_term = abs(error) - 1 / 2
    use_linear_term = (abs(error) > 1.0)
    if in_keras:
        # Keras won't let us multiply floats by booleans, so we explicitly cast the booleans to floats
        use_linear_term = K.cast(use_linear_term, 'float32')
    return use_linear_term * linear_term + (1 - use_linear_term) * quadratic_term


'''
def build_model(model_path, learning_rate=0.01):
    if os.path.exists(model_path):
        # model = load_model(model_path, custom_objects={'huber_loss': huber_loss})
        model = load_model(model_path)
        print('Load existing model.')
    else:
        model = Sequential()
        model.add(Dense(InputHandler.input_size * 2, input_shape=(InputHandler.input_size,), activation='relu',W_regularizer=l2(0.01)))
        model.add(Reshape((InputHandler.input_size * 2, 1,), input_shape=(InputHandler.input_size * 2,)))
        #model.add(Dense(InputHandler.input_size, input_shape=(InputHandler.input_size,), activation='relu',W_regularizer=l2(0.01)))
        model.add(Conv1D(filters=50, kernel_size=18, strides=18, padding='same', activation='relu'))
        model.add(Conv1D(filters=25, kernel_size=9, strides=9, padding='same', activation='relu'))
        model.add(Flatten())
        model.add(Dense(InputHandler.input_size * 2, activation='relu', W_regularizer=l2(0.01)))
        model.add(Dense(InputHandler.output_size, activation='linear'))
        # optimizer = RMSprop(lr=0.00025, rho=0.95, epsilon=0.01)
        optimizer = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
        # model.compile(loss=huber_loss, optimizer=optimizer)
        model.compile(loss=mean_squared_error, optimizer=optimizer)
        print('Create new model.')
    return model

'''


# TODO: first 2 Conv1D then 2 Fully
def build_model(model_path, learning_rate=0.01):
    if os.path.exists(model_path):
        # model = load_model(model_path, custom_objects={'huber_loss': huber_loss})
        model = load_model(model_path)
        print('Load existing model.')
    else:
        model = Sequential()
        model.add(Dense(InputHandler.input_size * 2, input_shape=(InputHandler.input_size,), activation='relu',W_regularizer=l2(0.01)))
        model.add(Reshape((InputHandler.input_size * 2, 1,), input_shape=(InputHandler.input_size * 2,)))
        #model.add(Dense(InputHandler.input_size, input_shape=(InputHandler.input_size,), activation='relu',W_regularizer=l2(0.01)))
        model.add(Conv1D(filters=50, kernel_size=9, strides=9, padding='same', activation='relu'))
        model.add(Conv1D(filters=50, kernel_size=18, strides=9, padding='same', activation='relu'))
        model.add(Conv1D(filters=50, kernel_size=36, strides=9, padding='same', activation='relu'))
        model.add(Conv1D(filters=25, kernel_size=9, strides=9, padding='same', activation='relu'))
        model.add(Flatten())
        model.add(Dense(InputHandler.input_size * 2, activation='relu', W_regularizer=l2(0.01)))
        model.add(Dense(InputHandler.output_size, activation='linear'))
        # optimizer = RMSprop(lr=0.00025, rho=0.95, epsilon=0.01)
        optimizer = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
        # model.compile(loss=huber_loss, optimizer=optimizer)
        model.compile(loss=mean_squared_error, optimizer=optimizer)
        print('Create new model.')
    return model
