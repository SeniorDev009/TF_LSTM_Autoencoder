from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import bisect
import os
import librosa
from tqdm import tqdm
import random
import re


def get_anom_score(data, response, smooth_param):
    anom_score = (response - data)
    anom_score = smooth(anom_score, smooth_param)
    anom_score = (anom_score - np.mean(anom_score)) / np.std(anom_score)
    anom_score = abs(anom_score)
    return anom_score


def gen_model(window_width, LAYER_SIZE, HIDDEN_LAYER_COUNT):
    model = keras.Sequential()
    model.add(keras.layers.LSTM(LAYER_SIZE, input_shape=(
        window_width, 1), return_sequences=True))
    for i in range(HIDDEN_LAYER_COUNT):
        model.add(keras.layers.LSTM(LAYER_SIZE, return_sequences=True))
    model.add(keras.layers.Dense(1))
    model.summary()
    model.compile(loss='mae', optimizer='adam')
    return model


def get_all_files(path):  # sr is sample rate
    files = []
    i = 0
    # r=root, d=directories, f = files
    for r, d, f in os.walk(path):
        for file in f:
            if '.wav' in file and i % 1 == 0:
                files.append(os.path.join(r, file))
            i = i + 1
    random.shuffle(files)
    return files


def get_data_from_files(filenames, sr, ww, step=None, file_length=30, nrm_type='EACH'):
    if step is None:
        step = ww
    file_multiplier = (file_length*sr-ww)//step+1
    result = np.zeros((len(filenames)*file_multiplier, ww))
    for ind, i in tqdm(enumerate(filenames)):
        raw_data, _ = librosa.load(i, sr=sr)
        if nrm_type == 'ALL':
            raw_data = (raw_data-np.mean(raw_data)) / \
                np.std(raw_data)  # normalization
        data = rolling(raw_data, ww, step)
        if nrm_type == 'EACH':
            meanraw = np.mean(data, axis=1)
            stdraw = np.std(data, axis=1)
            data = (data - meanraw[:, np.newaxis]) / stdraw[:, np.newaxis]
        result[ind*file_multiplier:(ind+1)*file_multiplier, :] = data
    return result


def scaleData(data):
    # normalize features
    data = data.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    conv_data = scaler.fit_transform(data)
    conv_data = np.squeeze(conv_data)
    return conv_data


def get_anomaly_mask(timestamps, labels):
    mask = np.zeros(len(timestamps))
    for i in labels:
        startp = bisect.bisect_left(timestamps, i[0])
        endp = bisect.bisect_right(timestamps, i[1])
        mask[startp:endp] = 1
    return mask


def rolling(a, window, step):
    shape = (a.size - window + 1, window)
    strides = a.strides * 2
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)[0::step]


def unroll(data, rolling_step):
    len = data.shape[1] + rolling_step * (data.shape[0] - 1)
    ans = np.zeros(len)
    cnt = np.zeros(len)
    for i in range(data.shape[0]):
        ans[i * rolling_step:i * rolling_step + data.shape[1]] += data[i, :]
        cnt[i * rolling_step:i * rolling_step + data.shape[1]] += 1
    ans = np.divide(ans, cnt)
    return ans


def xcorr(x, y, maxlags):
    Nx = len(x)
    if Nx != len(y):
        raise ValueError('x and y must be equal length')
    c = np.correlate(x, y, mode=2)
    if maxlags is None:
        maxlags = Nx - 1
    if maxlags >= Nx or maxlags < 1:
        raise ValueError('maxlags must be None or strictly positive < %d' % Nx)
    c = c[Nx - 1 - maxlags:Nx + maxlags]

    return c


def smooth(x, N):
    if N < 2:
        return x
    else:
        return np.convolve(x, np.ones((N,)) / N, mode='same')


def dialog_choose_file(folder, mask):
    available_files = []
    r = re.compile(mask)
    for file in os.listdir(folder):
        if r.match(file):
            available_files.append(file)

    for ind, fl in enumerate(available_files):
        print(str(ind + 1) + ": " + fl)
    nb = int(input("Choose response file:"))

    if nb > len(available_files) or nb < 0:
        raise Exception(
            'number should be between 1 and {}'.format(len(available_files)))

    cur_file = folder + '/' + available_files[nb - 1]
    return cur_file
