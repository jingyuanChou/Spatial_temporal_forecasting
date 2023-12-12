from layer import *
import pickle

import matplotlib.pyplot as plt
import numpy as np
from idtxl.multivariate_te import MultivariateTE
from idtxl.bivariate_te import BivariateTE
from idtxl.data import Data
import pandas as pd
import networkx as nx
import torch
import multiprocessing


class MTE_STJGC(nn.Module):
    def __init__(self, batch_size, kernel_size, dilation_rates, num_nodes, horizon,
                 node_dim=16, seq_length=12, in_dim=1, layers=4, alpha=1.0, MTE=True):
        super(MTE_STJGC, self).__init__()
        self.batch_size = batch_size
        self.MTE = MTE
        self.num_nodes = num_nodes
        self.alpha = alpha
        self.horizon = horizon
        self.layers = layers
        self.skip_convs = nn.ModuleList()
        self.norm = nn.ModuleList()
        self.feature_transform = nn.Linear(in_dim, node_dim)
        self.softmax = nn.Softmax()
        self.relu = nn.ReLU()
        self.multi_interaction = nn.Embedding(node_dim, node_dim)
        self.seq_length = seq_length
        self.kernel_size = kernel_size
        self.dilation_rates = dilation_rates
        self.STJGC = DilatedSpatioTemporalGCN(node_dim, self.kernel_size, self.dilation_rates)

        self.output_module = nn.ModuleList()
        for _ in range(self.horizon):
            self.output_module.append(nn.Linear(node_dim, 1))

    def forward(self, input, original_data_for_MTE, batch_index):

        seq_len = input.size(0)
        assert seq_len == self.seq_length, 'input sequence length not equal to preset sequence length'
        '''
        # Perform MTE first
        MTE_data = original_data_for_MTE[:batch_index + 1, :]
        data_idtxl = Data(MTE_data, dim_order='ps',
                          normalise=False)  # use readily available data to calculate transfer entropy

        # Initialize the MultivariateTE analysis object
        network_analysis = MultivariateTE()
        # We should be able to check multiple timestamps, and record the transfer entropy of each [i,j] pair
        # Set some parameters
        settings = {'cmi_estimator': 'JidtKraskovCMI',
                    'max_lag_sources': seq_len,
                    'min_lag_sources': 0,
                    'verbose': False,
                    'parallel_target': 'cpu',
                    'parallel_params': 15}

        # Run the analysis
        results = network_analysis.analyse_network(settings=settings, data=data_idtxl)

        MTE_static_matrices = torch.zeros((self.seq_length, self.num_nodes, self.num_nodes))

        for target in results.keys():
            selected_sources = results[target]['selected_vars_sources']
            selected_sources_te = results[target]['selected_sources_te']
            selected_target_past = results[target]['selected_vars_target']
            if len(selected_sources) != 0:
                for idx, source in enumerate(selected_sources):
                    source_process = source[0]
                    source_process_lag = source[1]
                    source_te = selected_sources_te[idx]
                    MTE_static_matrices[source_process_lag - 1, source_process, target] = source_te
            if len(selected_target_past) != 0:
                for idx2, target_past in enumerate(selected_target_past):
                    target_p = target_past[0]
                    target_process_lag = target_past[1]
                    MTE_static_matrices[target_process_lag - 1, target_p, target] = 1
        
        MTE_static_matrices = self.softmax(MTE_static_matrices)
        
        MTE_static_matrices = torch.transpose(MTE_static_matrices, 1, 2, 0)
        '''
        MTE_static_matrices = torch.normal(1,1,(12,49,49))
        # Then obtain the embedding
        embedding_U = self.feature_transform(input)  # expand to latent space

        # Forward pass through the model
        output = self.STJGC(embedding_U, self.multi_interaction, MTE_static_matrices, self.alpha)

        # OUTPUT Predictions
        res_list = []
        for idx in range(self.horizon):
            res_list.append(self.output_module[idx](output))
        res_list = torch.stack(res_list, dim=1).squeeze()
        return res_list
