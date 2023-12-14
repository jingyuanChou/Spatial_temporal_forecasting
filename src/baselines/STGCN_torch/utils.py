import os
import zipfile

import networkx as nx
import numpy as np
import torch


def load_metr_la_data():
    if (not os.path.isfile("data/adj_mat.npy")
            or not os.path.isfile("data/node_values.npy")):
        with zipfile.ZipFile("data/METR-LA.zip", 'r') as zip_ref:
            zip_ref.extractall("data/")

    A = np.load("data/adj_mat.npy")
    X = np.load("data/node_values.npy").transpose((1, 2, 0))
    X = X.astype(np.float32)

    # Normalization using Z-score method
    means = np.mean(X, axis=(0, 2))
    X = X - means.reshape(1, -1, 1)
    stds = np.std(X, axis=(0, 2))
    X = X / stds.reshape(1, -1, 1)

    return A, X, means, stds

def load_hosp_data():
    import pandas as pd
    vel = pd.read_csv("C:\\Users\\6\\PycharmProjects\\FORECASTING_GNN\\benchmark data\\HOSP\\hosp_weekly_filt_case_data.csv")
    hospitalization = vel.reset_index()
    hospitalization['state'] = hospitalization['state'].astype(int)
    neighbouring_states = pd.read_csv(
        'C:\\Users\\6\\PycharmProjects\\FORECASTING_GNN\\benchmark data\\HOSP\\neighbouring_states.csv')
    fips_states = pd.read_csv('C:\\Users\\6\\PycharmProjects\\FORECASTING_GNN\\benchmark data\\HOSP\\state_hhs_map.csv',
                              header=None).iloc[:(-3), :]
    fips_2_state = fips_states.set_index(0)[2].to_dict()
    hospitalization['state'] = hospitalization['state'].map(fips_2_state)
    state_2_index = hospitalization.set_index('state')['index'].to_dict()
    neighbouring_states['StateCode'] = neighbouring_states['StateCode'].map(state_2_index)
    neighbouring_states['NeighborStateCode'] = neighbouring_states['NeighborStateCode'].map(state_2_index)
    hospitalization = hospitalization.iloc[:, 30:]  # remove all 0 datapoints
    hospitalization = hospitalization.T.values

    G = nx.from_pandas_edgelist(neighbouring_states, 'StateCode', 'NeighborStateCode')
    adj = nx.adjacency_matrix(G)
    A = torch.from_numpy(adj.A)

    X = hospitalization
    X = torch.tensor(X).transpose(1,0)
    X = X.unsqueeze(1)
    X = X.float()
    # Normalization using Z-score method
    means = torch.mean(X, axis=(0, 2))
    X = X - means.reshape(1, -1, 1)
    stds = torch.std(X, axis=(0, 2))
    X = X / stds.reshape(1, -1, 1)

    return A, X, means, stds


def data_transform(data, n_his, n_pred, device):
    # produce data slices for x_data and y_data

    n_vertex = data.shape[1]
    len_record = len(data)
    num = len_record - n_his - n_pred

    x = np.zeros([num, 1, n_his, n_vertex])
    y = np.zeros([num, 1, n_pred, n_vertex])

    for i in range(num):
        head = i
        tail = i + n_his
        x[i, :, :, :] = data[head: tail].reshape(1, n_his, n_vertex)
        y[i, :, :, :] = data[tail: (tail + n_pred)].reshape(1, n_pred, n_vertex)

    return torch.Tensor(x).to(device), torch.Tensor(y).to(device)

def get_normalized_adj(A):
    """
    Returns the degree normalized adjacency matrix.
    """
    A = A + np.diag(np.ones(A.shape[0], dtype=np.float32))
    D = np.array(torch.sum(A, axis=1)).reshape((-1,))
    D[D <= 10e-5] = 10e-5    # Prevent infs
    diag = np.reciprocal(np.sqrt(D))
    A_wave = np.multiply(np.multiply(diag.reshape((-1, 1)), A),
                         diag.reshape((1, -1)))
    return A_wave


def generate_dataset(X, num_timesteps_input, num_timesteps_output):
    """
    Takes node features for the graph and divides them into multiple samples
    along the time-axis by sliding a window of size (num_timesteps_input+
    num_timesteps_output) across it in steps of 1.
    :param X: Node features of shape (num_vertices, num_features,
    num_timesteps)
    :return:
        - Node features divided into multiple samples. Shape is
          (num_samples, num_vertices, num_features, num_timesteps_input).
        - Node targets for the samples. Shape is
          (num_samples, num_vertices, num_features, num_timesteps_output).
    """
    # Generate the beginning index and the ending index of a sample, which
    # contains (num_points_for_training + num_points_for_predicting) points
    indices = [(i, i + (num_timesteps_input + num_timesteps_output)) for i
               in range(X.shape[2] - (
                num_timesteps_input + num_timesteps_output) + 1)]

    # Save samples
    features, target = [], []
    for i, j in indices:
        temp = X[:, :, i: (i + num_timesteps_input)]
        temp = torch.transpose(temp,2,1)
        t = X[:, 0, i + num_timesteps_input: (j)]
        if t.shape[1] != 4:
            break
        features.append(temp)
        target.append(t)

    features = torch.stack(features, dim=0)
    target = torch.stack(target, dim=0)
    return features, target
