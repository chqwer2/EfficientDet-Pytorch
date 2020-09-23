#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (c) 2020, ASML Netherlands B.V.
#
# THIS SOFTWARE, IN SOURCE CODE, OBJECT CODE  AND SCRIPT FORM, IS THE PROPRIETARY AND
# CONFIDENTIAL INFORMATION OF ASML NETHERLANDS B.V. (AND ITS AFFILIATES) AND IS
# PROTECTED BY U.S. AND INTERNATIONAL LAW.  ANY UNAUTHORIZED USE, COPYING AND
# DISTRIBUTION OF THIS SOFTWARE, IN SOURCE CODE, OBJECT CODE AND SCRIPT FORM, IS
# STRICTLY PROHIBITED. THIS SOFTWARE, IN SOURCE CODE, OBJECT CODE AND SCRIPT FORM IS
# PROVIDED ON AN "AS IS" BASIS WITHOUT WARRANTY, EXPRESS OR IMPLIED.  ASML
# NETHERLANDS B.V. EXPRESSLY DISCLAIMS THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
# FITNESS FOR A PARTICULAR PURPOSE AND ASSUMES NO RESPONSIBILITY FOR ANY ERRORS
# THAT MAY BE INCLUDED IN THIS SOFTWARE, IN SOURCE CODE, OBJECT CODE AND IN SCRIPT
# FORM. ASML NETHERLANDS B.V. RESERVES THE RIGHT TO MAKE CHANGES TO THE SOFTWARE, IN
# SOURCE CODE, OBJECT CODE  AND SCRIPT FORM WITHOUT NOTICE.

"""
Created on Thu Feb  6 10:59:23 2020

@author: chenlin
"""
# import required funtions from Keras
# from numba import autojit, jit
import numpy as np
import pandas as pd
# to support plot using gpu queue
import matplotlib as mpl
mpl.use('Agg')

# %matplotlib inline
import os
import sys

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import GRU, Embedding, Dense, AveragePooling2D, PReLU, LeakyReLU,\
                         Activation, Flatten, Reshape, concatenate, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam, SGD, Nadam, Adadelta


# from tensorflow.keras.layers.wrappers import Bidirectional
from tensorflow.keras.applications import vgg19
import logging
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras import layers
from tensorflow.keras.models import Model, load_model

# to support plot using gpu queue
from utils.cfg import Config
from sklearn.externals import joblib
import hybrid_loss as hdl
from octConv import OctConv
from ACNet import ACnet


# hyper parameters
epoch = 80
lr=0.002       #0.01 too big
Batch_Size=512
mu_tau_weight = 0.80
standard_mutau = False
LoadWeights = False
model_dict = './cnn_result/20200916_17-38-27epoch100_simple_net_lr0.001_FitStandardScalerFromAllMutauFalse_Z5_order0/checkpoint-0100'
train_partially = False    #Not using all data set in order to get higher speed
P = 0.3


cfg_file = './cfg.txt'
cfg = Config.init_from_cfg_file(cfg_file)


def _simple_net():
    input_img = tf.keras.Input(shape=(128, 128, 1))  # Input placeholder

    x = input_img
    for channel_num in [4, 8, 16, 32, 64]:  #32, 128
        # x = layers.ZeroPadding2D((1, 1))(x)
        x = layers.Conv2D(channel_num, (3, 3),  strides=(1, 1),padding='same', kernel_initializer='glorot_uniform',
                          kernel_regularizer=tf.keras.regularizers.l2(0.00001))(x)
        # x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.12)(x)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)

    for channel_num in [128]:  # 32, 128
        x = layers.Conv2D(channel_num, (3, 3), strides=(1, 1), padding='same', kernel_initializer='glorot_uniform',
                          kernel_regularizer=tf.keras.regularizers.l2(0.00001))(x)
        x = LeakyReLU(alpha=0.12)(x)


    x = layers.Flatten()(x)

    x = layers.Dense(1024, activation='relu', kernel_initializer='glorot_uniform')(x)
    x = layers.Dense(512, activation='relu', kernel_initializer='glorot_uniform')(x)   #change 'linear'
    x = layers.Dense(4, activation='linear', kernel_initializer='glorot_uniform')(x)
    model = Model(inputs=input_img, outputs=x, name='simple_net')
    return model

def _curent_time():
    from datetime import datetime
    date = datetime.now()
    return date.strftime("%Y%m%d_%H-%M-%S")


from sklearn.preprocessing import StandardScaler, MinMaxScaler
global scalar
scalar = StandardScaler()
# scalar = MinMaxScaler(copy=True, feature_range=(-1, 1))



def _prepare_data(order, stanard_all_mutau):
    IMAGE_DIR = '/nfs/DEV/PWCGUI/willchen/codes/ml_lens_model_train/tools/large_data_train/imgs_preprocess/imgs_npy_compresszed_float32.npz'
    dp_imgs = np.load(IMAGE_DIR)
    dp_imgs = dp_imgs['arr_0']

    mutau_path = "/nfs/DEV/PWCGUI/willchen/codes/ml_lens_model_train/tools/large_data_train/mutau_npy/Z5_pred_target.npy"
    mutau = np.load(mutau_path, allow_pickle=True)
    mutau = mutau[:,1]
    mutau_order = np.zeros((len(mutau), 4))
    for idx, e in enumerate(mutau):
        mutau_order[idx] = e[0][order * 4: order * 4 + 4]
    #mutau_order = mutau[:, order * 4: order * 4 + 4]
    train_ID_DIR = '/nfs/DEV/PWCGUI/willchen/codes/ml_lens_model_train/tools/large_data_train/training_data_idx_in_large_data.npy'
    test_ID_DIR = '/nfs/DEV/PWCGUI/willchen/codes/ml_lens_model_train/tools/large_data_train/test_data_idx_in_large_data.npy'
    DP_LIST_FILE = '/nfs/DEV/PWCGUI/willchen/codes/ml_lens_model_train/tools/large_data_train/mutau_npy/dp_list.csv'
    train_ids = np.load(train_ID_DIR)   #[     0      1      2 ... 176477 176478 176479]
    test_ids = np.load(test_ID_DIR)
    train_ids = np.append(train_ids, 45723)    #171273
    # ID = [45723, 144531, 119129, 135329]
    # for i in Wrong_ID:
    #     train_ids = np.delete(train_ids, np.where(train_ids == i))


    if train_partially:
        L = np.math.floor(len(train_ids) * P)
        train_ids = np.random.choice(train_ids, L, replace=False)  #Random Sample

    if stanard_all_mutau:
        logging.info("fit standard scaler from all data")
        mutau_order = scalar.fit_transform(mutau_order)

        mutau_train = mutau_order[train_ids]
        mutau_test = mutau_order[test_ids]
    else:
        # fit scaler from training data, and then apply to test data
        logging.info("fit standard scaler from training data, and apply to test data")
        mutau_train = scalar.fit_transform(mutau_order[train_ids])
        mutau_test = scalar.transform(mutau_order[test_ids])

    return dp_imgs[train_ids], mutau_train, dp_imgs[test_ids], mutau_test, scalar

def _plot_history(his_file, result_dir):
    if not os.path.exists(his_file):
        logging.error('trainning.log not exists to train')
    df = pd.read_csv(his_file)
    # loss_and_mae = df.filter(items=['loss', 'mae'])
    loss_and_mae = df.drop('epoch', 1)

    # TODO: a lot of warnings
    def _draw():
        tmp_glob = {'loss_and_mae': loss_and_mae,
                    'output_path': os.path.join(result_dir, 'training_trend.png')}
        tmp_namespace = {}
        src = "ax = loss_and_mae.plot.line(grid=True);fig = ax.get_figure();fig.savefig(output_path)"
        exec(src, tmp_glob, tmp_namespace)

    _draw()

def _init_logger(log_file):
    logger = logging.getLogger()
    logger.setLevel('DEBUG') # need, or nothing happens

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel('DEBUG')

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel('DEBUG')

    #logging.basicConfig(filename=os.path.join(log_dir, 'log_{}.txt'.format(_curent_time())), filemode='a')
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger

class ModelPrediction():
    def __init__(self, model_path=None, model=None, standard_file=None):
        if model_path:
            self._model = keras.models.load_model(model_path)
        if model:
            self._model = model

        self._standard_scaler = joblib.load(standard_file) if standard_file else None

    def predict(self, input_imgs):
        mutau_npy = self._model.predict(input_imgs)
        return self._standard_scaler.inverse_transform(mutau_npy)


from tensorflow.python.ops import math_ops
from tensorflow.python.framework import ops
from tensorflow.python.keras import backend as K

def SmoothL1(true, pred, delta=9.0):
    y_pred = ops.convert_to_tensor(pred)
    y_true = math_ops.cast(true, y_pred.dtype)

    diff = tf.abs(y_true - y_pred)
    # condition, x = None, y = None
    smoothl1_loss = tf.where(
        tf.less(diff, 1.0 / delta),   #modified smoothL1
        0.5 * delta * tf.pow(diff, 2),
        diff - 0.5 / delta
    )

    return K.mean(smoothl1_loss, axis=-1)

def standard_huber(true, pred, delta=1.0):
    y_pred = ops.convert_to_tensor(pred)
    y_true = math_ops.cast(true, y_pred.dtype)

    diff = tf.abs(y_true - y_pred)

    huber_loss = tf.where(
        tf.less(diff, delta),  # modified smoothL1
        0.5  * tf.pow(diff, 2),
        delta * diff - 0.5 / tf.pow(delta, 2)
    )
    return K.mean(huber_loss, axis=-1)


def DEV_mse_loss_hybrid(ref, pred):  # feature, label?
    # 0.85    0.9994
    # cost1 = tf.keras.losses.MSE(ref, pred)  # MSE
    cost1 = SmoothL1(ref, pred, delta=9.0)
    # tf.keras.losses.logcosh(ref, pred)

    LHFF_pred, LHFF_true = hdl.mu_tau_to_LHFF(pred, ref, scalar)
    cost2 = SmoothL1(LHFF_pred, LHFF_true, delta=9.0) # Wrong
    # tf.print(" cost2:", cost2)
    result = mu_tau_weight * cost1 + (1 - mu_tau_weight) * cost2

    return result

def callbacks_(output_dir):

    training_log = os.path.join(output_dir, 'training.log')
    csv_logger = tf.keras.callbacks.CSVLogger(training_log)

    reduceLR = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=5, mode='min', verbose=1, min_delta=1e-3,
                                 cooldown=5,min_lr=1e-7)  # for bigger epoch

    # EarlyStop = EarlyStopping(monitor='loss', patience=10, verbose=1, min_delta=1e-6, mode='min', )  # baseline=0.08
    model_save_dict = os.path.join(output_dir, 'checkpoint-{epoch:04d}')
    modelcheckpoint = ModelCheckpoint(filepath=model_save_dict, verbose=1, save_weights_only=True, period=20)

    return [csv_logger, reduceLR, modelcheckpoint], training_log

models = {
    "SimpleNet":  _simple_net,
    "ACNet":  ACnet,
    "OctConv":  OctConv
}


def train():

    output_dir = './cnn_result/{}epoch{}_simple_net_lr{}_FitStandardScalerFromAllMutau{}_Z5_order0'.format(_curent_time(),epoch, lr, standard_mutau)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    _init_logger(os.path.join(output_dir, 'log.txt'))
    logging.info('epoch:{}'.format(epoch))

    model = _simple_net()
    # model = models[model_name]()
    # model = OctConv()
    # model = ACnet()


    if LoadWeights:
        logging.info("Loading weight form ", model_dict)
        model.load_weights(model_dict)

    logging.info('using lr:{}'.format(lr))
    logging.info(model.summary())

    adam = Adam(learning_rate=lr, amsgrad=False)
    nadam = Nadam(learning_rate=lr)
    sgd = SGD(learning_rate=1, momentum=0.99, nesterov=True)  #bigger LR, Batch = 64
    adadelta =  Adadelta( learning_rate=0.001, rho=0.95, epsilon=1e-7)

    #loss = cost, cost1, cost2 = hdl.DEV_mse_loss_hybrid(LHNetworkResult, y, scaler, FLAGS.mu_tau_weight, w)
    #loss='mean_squared_error',

    imgs_train, mutau_train, imgs_test, mutau_test, scaler = _prepare_data(0, standard_mutau)

    model.compile(optimizer=adam, loss=DEV_mse_loss_hybrid, metrics=['mae'])

    callbacks, training_log = callbacks_(output_dir)  #setting callbacks

    model.fit(imgs_train, mutau_train, epochs=epoch,
              batch_size=Batch_Size, verbose=1, validation_split=0.005, callbacks=callbacks, shuffle=True, workers=4)

    model.save(os.path.join(output_dir, 'cnn_mutau_model_saved.h5'))
    model.save_weights( os.path.join(output_dir, 'cnn_mutau_weights_saved'))
    _plot_history(training_log, output_dir)

    #test part ...
    test_predict = model.predict(imgs_test)
    if scaler:
        test_predict = scaler.inverse_transform(test_predict)
        mutau_test = scaler.inverse_transform(mutau_test)
        joblib.dump(scaler, os.path.join(output_dir, 'standard_scaler.save'))
    np.save(os.path.join(output_dir, 'y_pred_test.npy'), test_predict)
    np.save(os.path.join(output_dir, 'y_truth_test.npy'), mutau_test)

    """
    preds = model.evaluate(X_test, Y_test)
    print ("Loss = " + str(preds[0]))
    print ("Test Accuracy = " + str(preds[1]))
    """
    print("==output dir is ", output_dir)
    print('bash ../evaluate/eval_cnn_keras_mutau.sh ', output_dir)
    print('python Result_Showup.py -result_dir ', output_dir)

def apply(model_path):
    model = ModelPrediction(model_path)
    model.predict()

def abort(msg):
    sys.stderr.write("Error: {}\n".format(msg))
    sys.exit(-1)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-act_apply', type=str, required=False, help='model path')
    parser.add_argument('-act_train', type=str, default=1, required=False, help='')
    parser.add_argument('-dir_postfix', type=str, required=False, help='for train')
    opts = parser.parse_args()
    if all([opts.act_apply, opts.act_train]):
        abort('multiple action specified')
    if not any([opts.act_apply, opts.act_train]):
        abort('1 action is need at least')

    if opts.act_apply:
        apply(opts.act_apply)
    elif opts.act_train:
        train()

if __name__ == '__main__':
    main()
    print('=====Done====')