import numpy as np
import torch
from torch.autograd import Variable

class Consecutive_dataloader(object):
    def __init__(self, data, train, valid, device, horizon, window, prediction_window,normalize=2):
        self.P = window
        self.prediction_window = prediction_window
        self.h = horizon
        self.predict_data = None
        self.rawdat = data
        self.dat = np.zeros(self.rawdat.shape)
        self.n, self.m = self.dat.shape
        self.normalize = normalize
        self.scale = np.ones(self.m)
        self._normalized(normalize)
        self._split(int(train * self.n), int((train + valid) * self.n), self.n)
        self.scale = torch.from_numpy(self.scale).float()

        self.scale = self.scale.to(device)
        self.scale = Variable(self.scale)
        self.device = device

    def _normalized(self, normalize):
        # normalized by the maximum value of entire matrix.

        if (normalize == 0):
            self.dat = self.rawdat

        if (normalize == 1):
            self.dat = self.rawdat / np.max(self.rawdat)

        # normlized by the maximum value of each row(sensor).
        if (normalize == 2):
            for i in range(self.m):
                self.scale[i] = np.max(np.abs(self.rawdat[:, i]))
                self.dat[:, i] = self.rawdat[:, i] / self.scale[i]

    def _split(self, train, valid, test):
        train_set = range(self.P + self.h - 1, train)
        valid_set = range(train, valid)
        testset = range(valid, test)
        self.train = self._batchify(train_set, self.h)
        self.valid = self._batchify(valid_set, self.h)
        self.testset = self._batchify(testset, self.h)


    def _batchify(self, idx_set, horizon):
        n = len(idx_set)
        X = torch.zeros((n, self.P, self.m))
        Y = torch.zeros((n, self.h,self.m))
        if idx_set[0] == self.P + self.h - 1:
            data_split = 'train'
        elif idx_set[-1] == self.n - 1:
            data_split = 'testing'
        else:
            data_split = 'validation'
        for i in range(n):
            end = idx_set[i] - self.h + 1
            start = end - self.P
            X[i, :, :] = torch.from_numpy(self.dat[start:end, :])
            Y[i, :, :] = torch.from_numpy(self.dat[end:idx_set[i]+1, :])
        if data_split == 'train':
            training_index_MTE = range(idx_set[0]-self.h, idx_set[-1]-self.h+1)
            self.training_index_MTE = training_index_MTE
        elif data_split == 'validation':
            validation_index_MTE = range(idx_set[0]-self.h, idx_set[-1]-self.h+1)
            self.validation_index_MTE = validation_index_MTE
        else:
            test_index_MTE = range(idx_set[0]-self.h, idx_set[-1]-self.h+1)
            self.testing_index_MTE = test_index_MTE

        return [X, Y]
    '''
    def get_batches(self, inputs, targets, batch_size, shuffle=True):
        length = len(inputs)
        if shuffle:
            index = torch.randperm(length)
        else:
            index = torch.LongTensor(range(length))
        start_idx = 0
        while (start_idx < length):
            end_idx = min(length, start_idx + batch_size)
            excerpt = index[start_idx:end_idx]
            X = inputs[excerpt]
            Y = targets[excerpt]
            X = X.to(self.device)
            Y = Y.to(self.device)
            yield Variable(X), Variable(Y)
            start_idx += batch_size
    '''

    def get_batches(self, inputs, targets, batch_size):
        length = len(inputs)
        start_idx = 0
        while (start_idx < length):
            X = inputs[start_idx]
            Y = targets[start_idx]
            X = X.to(self.device)
            Y = Y.to(self.device)
            yield Variable(X), Variable(Y)
            start_idx += batch_size

def mape_loss(output, target):
    """
    Calculate Mean Absolute Percentage Error (MAPE) loss

    Args:
    output (torch.Tensor): The predictions
    target (torch.Tensor): The ground truth values

    Returns:
    torch.Tensor: MAPE loss
    """
    loss = torch.abs((target - output) / (target+1))
    return torch.mean(loss)




