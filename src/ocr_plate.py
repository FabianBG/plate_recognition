import keras
import tensorflow as tf
import numpy as np
print('TensorFlow version:', tf.__version__)
print('Keras version:', keras.__version__)
from PIL import Image, ImageDraw, ImageFont
from random import randint

import os
import sys
import itertools
import codecs
import re
import datetime
import cairocffi as cairo
import editdistance
from scipy import ndimage

from keras import backend as K
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Input, Dense, Activation
from keras.layers import Reshape, Lambda
from keras.layers.merge import add, concatenate
from keras.models import Model
from keras.layers.recurrent import GRU
from keras.optimizers import SGD
from keras.utils.data_utils import get_file
from keras.preprocessing import image
import keras.callbacks
from src.config import *


def rounded_rectangle(self: ImageDraw, xy, corner_radius, fill=None, outline=None):
    upper_left_point = xy[0]
    bottom_right_point = xy[1]
    self.rectangle(
        [
            (upper_left_point[0], upper_left_point[1] + corner_radius),
            (bottom_right_point[0], bottom_right_point[1] - corner_radius)
        ],
        fill=fill,
        outline=outline
    )
    self.rectangle(
        [
            (upper_left_point[0] + corner_radius, upper_left_point[1]),
            (bottom_right_point[0] - corner_radius, bottom_right_point[1])
        ],
        fill=fill,
        outline=outline
    )
    self.pieslice([upper_left_point, (upper_left_point[0] + corner_radius * 2, upper_left_point[1] + corner_radius * 2)],
        180,
        270,
        fill=fill,
        outline=outline
    )
    self.pieslice([(bottom_right_point[0] - corner_radius * 2, bottom_right_point[1] - corner_radius * 2), bottom_right_point],
        0,
        90,
        fill=fill,
        outline=outline
    )
    self.pieslice([(upper_left_point[0], bottom_right_point[1] - corner_radius * 2), (upper_left_point[0] + corner_radius * 2, bottom_right_point[1])],
        90,
        180,
        fill=fill,
        outline=outline
    )
    self.pieslice([(bottom_right_point[0] - corner_radius * 2, upper_left_point[1]), (bottom_right_point[0], upper_left_point[1] + corner_radius * 2)],
        270,
        360,
        fill=fill,
        outline=outline
    )

def random_plate():
    size = randint(plate_min, plate_max)
    letter = ""
    number = ""
    for _ in range(size):
        letter = letter + chr(big_letters[randint(0, len(big_letters) - 1)])
        number = number + chr(digits[randint(0, len(digits) - 1)])
    return letter + "-" + number


def get_plate(size, empty=False, show_plate=False, save_plate=""):
    plate = random_plate()
    font = ImageFont.truetype(font_path, 30)
    img = Image.new('RGB', size, (0,0,0))
    draw = ImageDraw.Draw(img)
    rounded_rectangle(draw, ((0, 0), size), 20, fill=(255, 255, 255), outline=(255, 255, 255))
    draw.text((25,10), plate, font=font, fill=(0,0,0))
    if show_plate: img.show()
    if save_plate is not empty: img.save(os.path.join(save_plate, plate + '.jpg'))
    img = np.array(img.convert('L'))
    img = img.astype(np.float32) / 255
    img = np.expand_dims(img, 0)
    return img, plate



def image_generator(size, img_w, img_h, downsample_factor, show_image=False):
    if K.image_data_format() == 'channels_first':
        X_data = np.ones([size, 1, img_w, img_h])
    else:
        X_data = np.ones([size, img_w, img_h, 1])
    labels = np.ones([size, absolute_max_string_len])
    input_length = np.zeros([size, 1])
    label_length = np.zeros([size, 1])
    source_str = []
    for i in range(size):
        data, world = get_plate(image_size, True, show_plate=show_image)
        if K.image_data_format() == 'channels_first':
            X_data[i, 0, 0:img_w, :] = np.array(data).T
        else:
            X_data[i, 0:img_w, :, 0] = data[0, :, :].T
        labels[i] = np.ones(absolute_max_string_len) * -1
        labels[i, 0:len(world)] = text_to_labels(world)
        input_length[i] = img_w // downsample_factor - 2
        label_length[i] = [len(world)]
        source_str.append(world)
    inputs = {
        'input': X_data,
        'labels': labels,
        'input_length': input_length,
        'label_length': label_length,
        'source_str': source_str,
        }
    outputs = {'ctc': np.zeros([size])}
    return (inputs, outputs)


def labels_to_text(labels):
    ret = []
    for c in labels:
        if c == len(alphabet):  # CTC Blank
            ret.append("")
        else:
            ret.append(chr(alphabet[c]))
    return "".join(ret)
    

def text_to_labels(text):
    ret = []
    for char in text:
        ret.append(alphabet.index(ord(char)))
    return ret

def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


def model(input_shape, output_shape, absolute_max_string_len):
   # Input Parameters
    img_h = input_shape[1]
    img_w = input_shape[0]
    
    # Network parameters
    conv_filters = 16
    kernel_size = (3, 3)
    pool_size = 2
    time_dense_size = 32
    rnn_size = 512

    if K.image_data_format() == 'channels_first':
        input_shape = (1, img_w, img_h)
    else:
        input_shape = (img_w, img_h, 1)

    act = 'relu'
    input_data = Input(name='input', shape=input_shape, dtype='float32')
    inner = Conv2D(conv_filters, kernel_size, padding='same',
                   activation=act, kernel_initializer='he_normal',
                   name='conv1')(input_data)
    inner = MaxPooling2D(pool_size=(pool_size, pool_size), name='max1')(inner)
    inner = Conv2D(conv_filters, kernel_size, padding='same',
                   activation=act, kernel_initializer='he_normal',
                   name='conv2')(inner)
    inner = MaxPooling2D(pool_size=(pool_size, pool_size), name='max2')(inner)

    conv_to_rnn_dims = (img_w // (pool_size ** 2),
                        (img_h // (pool_size ** 2)) * conv_filters)
    inner = Reshape(target_shape=conv_to_rnn_dims, name='reshape')(inner)
    inner = Dense(time_dense_size, activation=act, name='dense1')(inner)
    gru_1 = GRU(rnn_size, return_sequences=True,
                kernel_initializer='he_normal', name='gru1')(inner)
    gru_1b = GRU(rnn_size, return_sequences=True,
                 go_backwards=True, kernel_initializer='he_normal',
                 name='gru1_b')(inner)
    gru1_merged = add([gru_1, gru_1b])
    gru_2 = GRU(rnn_size, return_sequences=True,
                kernel_initializer='he_normal', name='gru2')(gru1_merged)
    gru_2b = GRU(rnn_size, return_sequences=True, go_backwards=True,
                 kernel_initializer='he_normal', name='gru2_b')(gru1_merged)
    inner = Dense(output_shape, kernel_initializer='he_normal',
                  name='dense2')(concatenate([gru_2, gru_2b]))
    
    y_pred = Activation('softmax', name='softmax')(inner)
    Model(inputs=input_data, outputs=y_pred).summary()

    labels = Input(name='labels',
                   shape=[absolute_max_string_len], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')
    loss_out = Lambda(
        ctc_lambda_func, output_shape=(1,),
        name='ctc')([y_pred, labels, input_length, label_length])
    test_func = K.function([input_data], [y_pred])
    sgd = SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)
    model = Model(inputs=[input_data, labels, input_length, label_length],
                  outputs=[loss_out, y_pred])
    model.compile(loss={'ctc': ctc, 'ctc': ctc}, optimizer=sgd)

    return model, test_func

def ctc(y_true, y_pred): 
    return y_pred

def batch_generator(size):
    while 1:
        data = image_generator(size, img_w=200, img_h=50, downsample_factor=4)
        yield data

class VizCallback(keras.callbacks.Callback):

    def __init__(self):
        self.output_dir = os.path.join("plate_test")
    
    def on_epoch_end(self, epoch, logs={}):
        self.model.save_weights(
            os.path.join(self.output_dir, 'weights%02d.h5' % (epoch)))

def decode_batch(test_func, word_batch):
    out = test_func([word_batch])[0]
    ret = []
    for j in range(out.shape[0]):
        out_best = list(np.argmax(out[j, 2:], 1))
        out_best = [k for k, g in itertools.groupby(out_best)]
        outstr = labels_to_text(out_best)
        ret.append(outstr)
    return ret

