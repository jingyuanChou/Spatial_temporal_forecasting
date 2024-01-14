import pickle

import numpy as np
import torch

if __name__ == '__main__':

    for time in range(24, 91):
        with open('MTE/MTE_51_states_cases_from_0_to_{}'.format(str(time)),'rb') as file:
            results = pickle.load(file)

        MTE_static_matrices = torch.zeros((12, 51, 51))

        for target in results.keys():
            selected_sources = results[target]['selected_vars_sources']
            selected_sources_te = results[target]['selected_sources_te']
            selected_target_past = results[target]['selected_vars_target']
            if len(selected_sources) != 0:
                for idx, source in enumerate(selected_sources):
                    source_process = source[0]
                    source_process_lag = source[1]
                    source_te = selected_sources_te[idx]
                    if source_process_lag == 0:
                        MTE_static_matrices[
                            11, source_process, target] = source_te
                    else:
                        index_ls = list(range(0, source_process_lag)) # lags start from 1
                        selected_index = [11-ind-1 for ind in index_ls]
                        MTE_static_matrices[selected_index, source_process, target] = source_te # fix this one, should be all the stuff
            if len(selected_target_past) != 0:
                for idx2, target_past in enumerate(selected_target_past):
                    target_process = target_past[0]
                    target_process_lag = target_past[1]
                    if target_process_lag == 0:
                        MTE_static_matrices[
                            11, target_process, target] = 1
                    else:
                        index_ls = list(range(0, target_process_lag)) # lags start from 1
                        selected_index = [11-ind-1 for ind in index_ls]
                        MTE_static_matrices[selected_index, target_process, target] = 1

        MTE_static_matrices = MTE_static_matrices.numpy()

        np.save('MTE_matrices_from_0_to_{}.npy'.format(str(time)), MTE_static_matrices)
