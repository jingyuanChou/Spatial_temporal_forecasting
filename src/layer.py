from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import dense_to_sparse

class AttentionMechanism(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.W_a = nn.Parameter(torch.Tensor(feature_dim, feature_dim))
        self.v = nn.Parameter(torch.Tensor(feature_dim, 1))
        self.b_a = nn.Parameter(torch.Tensor(feature_dim))

        nn.init.xavier_uniform_(self.W_a)
        nn.init.xavier_uniform_(self.v)
        nn.init.zeros_(self.b_a)

    def forward(self, last_layer_output):
        # last_layer_output is of shape (N, d, M)

        # Apply weight matrix using einsum
        S = torch.einsum('ndm,dd->ndm', last_layer_output, self.W_a)

        # Adding the bias term b_a
        # b_a is unsqueezed to match the dimensions: (d,) -> (1, d, 1)
        S = S + self.b_a.unsqueeze(0).unsqueeze(-1)

        # Apply the tanh activation function
        S = torch.tanh(S)

        # Apply the vector 'v' and compute the attention scores
        S = torch.einsum('ndm,dm->nm', S, self.v)
        A = F.softmax(S, dim=1)  # Shape (N, M)

        # Weighted sum of the last_layer_output using the attention scores
        Y = torch.einsum('nm,ndm->nd', A, last_layer_output)  # Shape (N, d)
        return Y

class DilatedSpatioTemporalGCN(nn.Module):
    def __init__(self, node_features, kernel_size, dilation_rates):
        super().__init__()
        # GCN layer for spatial convolution only at the first layer
        self.dilation_rates = dilation_rates
        self.kernel_size = kernel_size
        # Temporal convolutional layers for subsequent layers with dilation
        self.temporal_convs = nn.ModuleList()
        for dilation in dilation_rates:
            # padding = (kernel_size - 1) * dilation  # Padding to keep the temporal length consistent
            # We perform padding during the for loop, initialize layers first
            self.temporal_convs.append(nn.Conv1d(node_features, node_features, kernel_size,
                                                 dilation=dilation))
        # Initialize GCN layers for each dilation rate
        self.gcn_layers = nn.ModuleList()
        for _ in dilation_rates:  # Subsequent layers
            self.gcn_layers.append(GCNConv(node_features, node_features))

        self.attention_mechanism = AttentionMechanism(node_features)

    def forward(self, node_embeddings, B, static_MTE_matrix, alpha):
        '''
        :param node_embeddings: node embeddings for N nodes
        :param B: the interactive matrix ,a d by d matrix, d is the latent dimension
        :param static_MTE_matrix: static MTE matrix from Multivariate Transfer Entropy
        :param alpha: the weight to control the matrix
        :return: the representation of N nodes at timestamp t from each STJCN layer
        '''
        # Initializations as before ...

        # Assume B is given or computed within this function
        # B would be the adjacency relationship matrix you mentioned
        res_aggregation = []
        for layer_idx, (gcn_layer, temporal_conv) in enumerate(zip(self.gcn_layers, self.temporal_convs)):
            # Placeholder for new GCN outputs for this layer
            residual = node_embeddings
            new_gcn_outputs = []

            # Perform GCN for each timestamp
            for time_step in range(len(node_embeddings)): # 12 timestamps
                # Compute the dynamic adjacency matrix for this timestamp
                # Here we assume U is node_embeddings[time_step] and U_t is node_embeddings[-1]
                #dynamic_adj_matrix = F.softmax(torch.matmul(torch.matmul(node_embeddings[time_step], B),
                #                                  node_embeddings[-1].T))
                dynamic_adj_matrix = F.softmax(F.relu(torch.matmul(torch.matmul(node_embeddings[time_step], B.weight),
                                                            node_embeddings[-1].T)), dim=0)
                # The static matrix only shows at the first layer
                if layer_idx == 0:
                    dynamic_adj_matrix = alpha * static_MTE_matrix[time_step] + (1-alpha) * dynamic_adj_matrix

                # Convert dynamic adjacency matrix to edge indices and weights
                edge_index, edge_weight = dense_to_sparse(dynamic_adj_matrix)

                # Perform GCN operation
                x_gcn = F.relu(gcn_layer(node_embeddings[time_step], edge_index, edge_weight))
                new_gcn_outputs.append(x_gcn)

            # Stack and transpose for temporal convolution: (batch_size, channels, length)
            x = torch.stack(new_gcn_outputs, dim=2)  # num_nodes * out_features * length

            # Apply padding only on the left
            left_padding = (self.kernel_size - 1) * self.dilation_rates[layer_idx]
            x_padded = F.pad(x, (left_padding, 0), 'constant', 0)

            # Apply dilated causal convolution
            x = F.relu(temporal_conv(x_padded))
            res_aggregation.append(x[:,:,-1])

            # Update node_embeddings for next iteration if not the last layer
            node_embeddings = [x[:, :, t] for t in range(x.size(2))]
            node_embeddings = torch.stack(node_embeddings, dim=0)
            node_embeddings = node_embeddings+ residual

        res_aggregation = torch.stack(res_aggregation, dim=2)
        # Apply attention mechanism to the output of the last temporal convolution layer
        Y = self.attention_mechanism(res_aggregation)

        return Y
