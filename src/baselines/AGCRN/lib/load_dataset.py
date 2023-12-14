import os
import numpy as np
import pandas as pd


def load_st_dataset(dataset):
    #output B, N, D
    if dataset == 'PEMSD4':
        data_path = os.path.join('C:\\Users\\6\\PycharmProjects\\FORECASTING_GNN\\baselines\\AGCRN\\data\\pems04.npz')
        data = np.load(data_path)['data'][:, :, 0]  #onley the first dimension, traffic flow data
    elif dataset == 'PEMSD8':
        data_path = os.path.join('C:\\Users\\6\\PycharmProjects\\FORECASTING_GNN\\baselines\\AGCRN\\data\\pems08.npz')
        data = np.load(data_path)['data'][:, :, 0]  #onley the first dimension, traffic flow data
    elif dataset == 'HOSP':
        hospitalization = pd.read_csv('C:\\Users\\6\\PycharmProjects\\FORECASTING_GNN\\hosp_weekly_filt_case_data.csv')
        hospitalization = hospitalization.reset_index()
        hospitalization['state'] = hospitalization['state'].astype(int)
        neighbouring_states = pd.read_csv('C:\\Users\\6\\PycharmProjects\\FORECASTING_GNN\\neighbouring_states.csv')
        fips_states = pd.read_csv('C:\\Users\\6\\PycharmProjects\\FORECASTING_GNN\\state_hhs_map.csv', header=None).iloc[:(-3), :]
        fips_2_state = fips_states.set_index(0)[2].to_dict()
        hospitalization['state'] = hospitalization['state'].map(fips_2_state)
        state_2_index = hospitalization.set_index('state')['index'].to_dict()
        neighbouring_states['StateCode'] = neighbouring_states['StateCode'].map(state_2_index)
        neighbouring_states['NeighborStateCode'] = neighbouring_states['NeighborStateCode'].map(state_2_index)
        hospitalization = hospitalization.iloc[:, 30:]  # remove all 0 datapoints
        hospitalization = hospitalization.T.values
        data = hospitalization
    else:
        raise ValueError
    if len(data.shape) == 2:
        data = np.expand_dims(data, axis=-1)
    print('Load %s Dataset shaped: ' % dataset, data.shape, data.max(), data.min(), data.mean(), np.median(data))
    return data
