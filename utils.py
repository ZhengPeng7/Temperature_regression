import cv2
import numpy as np
from math import floor, ceil
import os
import pandas as pd
import keras
from keras.applications.vgg16 import preprocess_input
import keras.backend as K


# Data utils


def read_image_and_K_from_dir(dir_path, idx_test=7):
    train_paths = []
    test_paths = []
    train_K = []
    test_K = []
    train_labels = []
    test_labels = []
    print('Reading \ttrain and test set \tfrom subject ', end='')
    for file in sorted(os.listdir(dir_path), key=lambda x: int(x.replace('subject', '').split('.')[0])):
        print(file.replace('subject', '').split('.')[0], end=' ')
        data = pd.read_csv(os.path.join(dir_path, file), index_col=None)
        if int(file.split('.')[0][7:]) != idx_test:
            train_paths += np.asarray(data['path'].dropna()).tolist()
            train_K += np.asarray(data['K'].dropna()).tolist()
            train_labels += np.asarray(data['temperature'].dropna()).tolist()
        else:
            test_paths += np.asarray(data['path'].dropna()).tolist()
            test_K += np.asarray(data['K'].dropna()).tolist()
            test_labels += np.asarray(data['temperature'].dropna()).tolist()
    print('.csv')
    return train_paths, train_labels, train_K, test_paths, test_labels, test_K


def generate_generator(img_paths, labels, K, batch_size=32, net='inceptionV4', reverse=0, with_K='no', K_len=None):
    flag_continue = 0
    idx_total = 0
    data_len = len(labels) if isinstance(labels, list) else labels.shape[0]
    while True:
        if not flag_continue:
            x = []
            y = []
            k = []
            inner_iter_num = batch_size
        else:
            idx_total = 0
            inner_iter_num = batch_size - data_len % batch_size
        for _ in range(inner_iter_num):
            if idx_total >= data_len:
                flag_continue = 1
                break
            else:
                flag_continue = 0
            img = cv2.cvtColor(cv2.imread(img_paths[idx_total]), cv2.COLOR_BGR2RGB)
            if with_K == 'head':
                # print((img.shape[0], img.shape[1], 1))
                # print('K:', K)
                # print('idx_total:', idx_total)
                # print(K[idx_total])
                img = np.concatenate((img, np.zeros((img.shape[0], img.shape[1], 1), dtype=np.float32)+K[idx_total]), axis=-1)
            elif with_K == 'no' or 'tail':
                pass
            x.append(img)
            y.append(labels[idx_total])
            k.append(K[idx_total])
            if reverse:
                pass
            idx_total += 1
        if not flag_continue:
            x, y = np.asarray(x).astype(np.float), np.asarray(y)
            # print(x.shape, y.shape)
            if with_K == 'tail':
#                 yield ({'img': x, 'K': k}, y)
                for i in range(len(k)):
                    k[i] = np.zeros((K_len,)) + k[i]
                k = np.asarray(k)
                yield [x, k], y
            else:
                yield x, y


def read_48_points_index(dir_points='./data/48_points'):
    subject_files = sorted(os.listdir(dir_points), key=lambda x: int(x.split('.')[0].strip('subject')))
    index = []
    for fil in subject_files:
        data_all = np.array(pd.read_table(os.path.join(dir_points, fil), header=None, index_col=None)).astype(np.float)
        index.append(data_all[:, 0].astype(int))
    return index


def read_48_points(dir_path='./data/csv_files', dir_points='./data/48_points', idx_test=7):
    indices = read_48_points_index(dir_points)
    K = [97.1443351260611, 69.5664501664216, 159.927906153263, 197.541604656728, 196.532785605028, 150.187591154692,
         73.9067564389404, 98.1376187257627, 87.6987186119752, 87.1398284317404, 208.579199065803, 66.8972390272240,
         269.173134394676, 95.9096979594339, 63.9771862511073, 90.5700816308852]
    train_paths = []
    test_paths = []
    train_K = []
    test_K = []
    train_labels = []
    test_labels = []
    csv_files = sorted(os.listdir(dir_path), key=lambda x: int(x.split('.')[0].strip('subject')))
    print('Reading \tvalidation set \t\tfrom subject ', end='')
    for idx_fil, file in enumerate(csv_files):
        idx_sensor = indices[idx_fil]
        data = pd.read_csv(os.path.join(dir_path, file), index_col=None)
        print(file.split('.')[0].strip('subject'), end=' ')
        if int(file.split('.')[0][7:]) != idx_test:
            train_paths.append(np.asarray(data['path'].dropna()).reshape(-1, 1)[idx_sensor])
            train_K.append(np.asarray(data['K'].dropna()).reshape(-1, 1)[idx_sensor])
            train_labels.append(np.asarray(data['temperature'].dropna()).reshape(-1, 1)[idx_sensor])
        else:
            test_paths.append(np.asarray(data['path'].dropna()).reshape(-1, 1)[idx_sensor])
            test_K.append(np.asarray(data['K'].dropna()).reshape(-1, 1)[idx_sensor])
            test_labels.append(np.asarray(data['temperature'].dropna()).reshape(-1, 1)[idx_sensor])
    print('.csv')
    train_paths = np.vstack(train_paths)
    test_paths = np.vstack(test_paths)
    train_K = np.vstack(train_K)
    test_K = np.vstack(test_K)
    train_labels = np.vstack(train_labels)
    test_labels = np.vstack(test_labels)
    # train_paths, train_labels, train_K, 
    if idx_test is None:
        test_paths, test_labels, test_K = train_paths, train_labels, train_K
    return test_paths, test_labels, test_K



# Model utils


def loss_APE(labels, preds):
    return K.mean(K.abs(preds - labels) / labels ,axis=1) * 100.


class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
    def on_train_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))


class SaveModelOnAPE(keras.callbacks.Callback):
    def on_epoch_end(self, batch, logs={}):
        self.APE = round(logs.get('val_loss'), 3)
        if self.APE < 12.:
            self.model.save(os.path.join('weights', self.model.name + '_APE' + str(self.APE) + '.hdf5'))


class SaveModelOnMAE_tail(keras.callbacks.Callback):
    def on_epoch_end(self, batch, logs={}):
        self.MAE = round(logs.get('val_loss'), 3)
        if self.MAE < (np.sqrt(5) - 1) / 2:
            self.model.save(os.path.join('weights', 'weights_tail_K', self.model.name + '_tail_K_MAE' + str(self.MAE) + '.hdf5'))


class SaveModelOnMAE_head(keras.callbacks.Callback):
    def on_epoch_end(self, batch, logs={}):
        self.MAE = round(logs.get('val_loss'), 3)
        if self.MAE < (np.sqrt(5) - 1) / 2:
            self.model.save(os.path.join('weights', 'weights_head_K', self.model.name + '_head_K_MAE' + str(self.MAE) + '.hdf5'))


class SaveModelOnMAE_no(keras.callbacks.Callback):
    def on_epoch_end(self, batch, logs={}):
        self.MAE = round(logs.get('val_loss'), 3)
        if self.MAE < (np.sqrt(5) - 1) / 2:
            self.model.save(os.path.join('weights', 'weights_no_K', self.model.name + '_no_K_MAE' + str(self.MAE) + '.hdf5'))
