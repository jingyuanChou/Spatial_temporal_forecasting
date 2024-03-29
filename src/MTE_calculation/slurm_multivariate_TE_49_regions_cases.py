import pickle
from idtxl.multivariate_te import MultivariateTE
from idtxl.data import Data
import pandas as pd
import os,sys

time=int(sys.argv[1])
if __name__ == '__main__':
    # 91 weeks of 51 states, from July 2020 to March 2022
    cases = pd.read_csv('weekly_filt_case_data_July2020_Mar2022.csv')
    cases = cases.iloc[:, 3:]  # remove first 3 columns, first is state name,second is abbreviation, third is FIPS code
    cases = cases.T.values # size: timestamps * num_instances
    total_timestamps = cases.shape[0]

    # we will loop from 6 as we select lag of 3

    print('=================================== Running '+str(time)+'-th timestamp, total  timestamps')
    # Arrange the data in a 2D array
    data = cases[0:time]
    data = data.T # now it's 51 by (time+12), number_states by available time sequence from 0

    # Convert this into an IDTxl Data object
    data_idtxl = Data(data, dim_order='ps', normalise=False) # use readily available data to calculate transfer entropy

    # Initialize the MultivariateTE analysis object
    network_analysis = MultivariateTE()

    # We should be able to check multiple timestamps, and record the transfer entropy of each [i,j] pair
    # Set some parameters
    #print(f'Number of cores available: {multiprocessing.cpu_count()}')

    settings = {'cmi_estimator': 'JidtKraskovCMI',
                'max_lag_sources': 3,
                'min_lag_sources': 1,
                'verbose':False}

    # Run the analysis
    results = network_analysis.analyse_network(settings=settings, data=data_idtxl)

    with open('output/MTE_51_states_cases_from_{}_to_{}'.format(str(0), str(time)), 'wb') as f:
        pickle.dump(results._single_target, f)


