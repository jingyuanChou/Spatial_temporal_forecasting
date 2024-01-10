import pickle
from idtxl.multivariate_te import MultivariateTE
from idtxl.data import Data
import pandas as pd
import networkx as nx
import torch
import multiprocessing

if __name__ == '__main__':
    # 91 weeks of 51 states, from July 2020 to March 2022
    cases = pd.read_csv('weekly_filt_case_state_wise_data.csv')
    cases = cases.iloc[:, 3:]  # remove first 3 columns, first is state name,second is abbreviation, third is FIPS code
    cases = cases.T.values # size: timestamps * num_instances
    total_timestamps = cases.shape[0]

    # As we are using 12 weeks to predict next 4 weeks, we will loop from 12 (maybe 13, correct me if wrong)

    for time in range(0, total_timestamps - 12):
        print('=================================== Running '+str(time)+'-th timestamp, total 85 timestamps')
        # Arrange the data in a 2D array
        data = cases[0:(time+12)]
        data = data.T # now it's 51 by (time+12), number_states by available time sequence from 0

        # Convert this into an IDTxl Data object
        data_idtxl = Data(data, dim_order='ps', normalise=False) # use readily available data to calculate transfer entropy

        # Initialize the MultivariateTE analysis object
        network_analysis = MultivariateTE()

        # We should be able to check multiple timestamps, and record the transfer entropy of each [i,j] pair
        # Set some parameters
        print(f'Number of cores available: {multiprocessing.cpu_count()}')

        settings = {'cmi_estimator': 'JidtKraskovCMI',
                    'max_lag_sources': 12, 
                    'min_lag_sources': 0,
                    'verbose':False,
                    'parallel_target': 'cpu',
                    'parallel_params': 15}

        # Run the analysis
        results = network_analysis.analyse_network(settings=settings, data=data_idtxl)

        with open('MTE_51_states_cases_from_{}_to_{}'.format(str(time), str(time+12)), 'wb') as f:
            pickle.dump(results._single_target, f)


