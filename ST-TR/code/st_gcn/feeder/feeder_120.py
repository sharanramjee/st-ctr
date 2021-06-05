# sys
import os
import sys
import numpy as np
import random
import pickle

# torch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms

# visualization
import time

# operation
from . import tools

small = False


class Feeder120(torch.utils.data.Dataset):
    """ Feeder for skeleton-based action recognition
    Arguments:
        data_path: the path to '.npy' data, the shape of data should be (N, C, T, V, M)
        label_path: the path to label
        random_choose: If true, randomly choose a portion of the input sequence
        random_shift: If true, randomly pad zeros at the begining or end of sequence
        window_size: The length of the output sequence
        normalization: If true, normalize input sequence
        debug: If true, only use the first 100 samples
    """

    def __init__(self,
                 data_path,
                 data_path_1,
                 label_path,
                 label_path_1,
                 random_choose=False,
                 random_shift=False,
                 random_move=False,
                 window_size=-1,
                 normalization=False,
                 mirroring=False,
                 debug=False,
                 mmap=True):
        self.debug = debug
        self.data_path = data_path
        self.data_path_1 = data_path_1
        self.label_path = label_path
        self.label_path_1 = label_path_1
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.normalization = normalization
        self.load_data(mmap)
        if normalization:
            self.get_mean_map()

    def load_data(self, mmap):
        # data: N C V T M

        # load label
        if '.pkl' in self.label_path:
            try:
                with open(self.label_path) as f:
                    self.sample_name, self.label = pickle.load(f)
                with open(self.label_path_1) as f:
                    self.sample_name_1, self.label_1 = pickle.load(f)
                self.sample_name=self.sample_name+self.sample_name_1
                self.label = self.label + self.label_1
                print("label", len(self.label))

                print(self.sample_name.size())
            except:
                # for pickle file from python2
                with open(self.label_path, 'rb') as f:
                    self.sample_name, self.label = pickle.load(
                        f, encoding='latin1')
                with open(self.label_path_1, 'rb') as f:
                    self.sample_name_1, self.label_1 = pickle.load(
                        f, encoding='latin1')
                self.sample_name=self.sample_name+self.sample_name_1
                self.label = self.label + self.label_1
                print("label", len(self.label))

                print(len(self.sample_name))


        # old label format
        elif '.npy' in self.label_path:
            self.label = list(np.load(self.label_path))
            self.label_1 = list(np.load(self.label_path))
            self.label=self.label+self.label_1
            print("label", self.label.size())
            self.sample_name = [str(i) for i in range(len(self.label))]
            self.sample_name_1 = [str(i) for i in range(len(self.label))]

        else:
            raise ValueError()

        # load data
        if mmap:
            self.data = np.load(self.data_path, mmap_mode='r')
            self.data_1 = np.load(self.data_path_1, mmap_mode='r')
            print("d1",self.data_1.shape)
            self.data=np.append(self.data,self.data_1, axis=0)
            print("data", self.data.shape)
        else:
            self.data = np.load(self.data_path)
            self.data_1 = np.load(self.data_path_1)

        if (small):
            self.index = [i for (i, n) in enumerate(self.sample_name) if
                          n[16:20] in ["A010"]]
            self.sample_name = [self.sample_name[i] for i in self.index]

            self.label = [self.label[i] for i in self.index]
            self.data = self.data[self.index]

        if self.debug:
            self.label = self.label[0:100]
            self.data = self.data[0:100]
            self.sample_name = self.sample_name[0:100]

        self.N, self.C, self.T, self.V, self.M = self.data.shape

    def get_mean_map(self):
        data = self.data
        N, C, T, V, M = data.shape
        self.mean_map = data.mean(
            axis=2, keepdims=True).mean(
            axis=4, keepdims=True).mean(axis=0)
        self.std_map = data.transpose((0, 2, 4, 1, 3)).reshape(
            (N * T * M, C * V)).std(axis=0).reshape((C, 1, V, 1))

    def __len__(self):
        return len(self.label)

    def __iter__(self):
        return self

    def __getitem__(self, index):
        # get data
        data_numpy = np.array(self.data[index])
        label = self.label[index]
        sample=self.sample_name[index]

        # normalization
        if self.normalization:
            data_numpy = (data_numpy - self.mean_map) / self.std_map

        # processing
        if self.random_shift:
            data_numpy = tools.random_shift(data_numpy)
        if self.random_choose:
            data_numpy = tools.random_choose(data_numpy, self.window_size)
        elif self.window_size > 0:
            data_numpy = tools.auto_pading(data_numpy, self.window_size)
        if self.random_move:
            data_numpy = tools.random_move(data_numpy)


        return data_numpy, label, sample

    def top_k(self, score, top_k):
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)

    def top_k_by_category(self, score, top_k):
        return tools.top_k_by_category(self.label, score, top_k)

    def calculate_recall_precision(self, score):
        return tools.calculate_recall_precision(self.label, score)

    def is_training(self, state):
        self.state = state


def test(data_path, label_path, vid=None):
    import matplotlib.pyplot as plt
    loader = torch.utils.data.DataLoader(
        dataset=Feeder(data_path, label_path),
        batch_size=64,
        shuffle=False,
        num_workers=2)

    if vid is not None:
        sample_name = loader.dataset.sample_name
        sample_id = [name.split('.')[0] for name in sample_name]
        index = sample_id.index(vid)
        data, label = loader.dataset[index]
        data = data.reshape((1,) + data.shape)

        # for batch_idx, (data, label) in enumerate(loader):
        N, C, T, V, M = data.shape

        plt.ion()
        fig = plt.figure()
        ax = fig.add_subplot(111)

        pose, = ax.plot(np.zeros(V * M), np.zeros(V * M), 'g^')
        ax.axis([-1, 1, -1, 1])

        for n in range(N):
            for t in range(T):
                x = data[n, 0, t, :, 0]
                y = data[n, 1, t, :, 0]
                z = data[n, 2, t, :, 0]
                pose.set_xdata(x)
                pose.set_ydata(y)
                fig.canvas.draw()
                plt.pause(1)


if __name__ == '__main__':
    data_path = "./data/NTU-RGB-D/xview/val_data.npy"
    label_path = "./data/NTU-RGB-D/xview/val_label.pkl"

    test(data_path, label_path, vid='S003C001P017R001A044')
