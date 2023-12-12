from __future__ import division

import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
import numbers
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import dense_to_sparse


class nconv(nn.Module):
    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, x, A):
        x = x.double()
        A = A.double()
        x = torch.einsum('ncwl,vw->ncvl', (x, A))
        return x.contiguous()


class dy_nconv(nn.Module):
    def __init__(self):
        super(dy_nconv, self).__init__()

    def forward(self, x, A):
        x = torch.einsum('ncvl,nvwl->ncwl', (x, A))
        return x.contiguous()


class linear(nn.Module):
    def __init__(self, c_in, c_out, bias=True):
        super(linear, self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=bias)

    def forward(self, x):
        return self.mlp(x)


class prop(nn.Module):
    def __init__(self, c_in, c_out, gdep, dropout, alpha):
        super(prop, self).__init__()
        self.nconv = nconv()
        self.mlp = linear(c_in, c_out)
        self.gdep = gdep
        self.dropout = dropout
        self.alpha = alpha

    def forward(self, x, adj):
        adj = adj + torch.eye(adj.size(0)).to(x.device)
        d = adj.sum(1)
        h = x
        dv = d
        a = adj / dv.view(-1, 1)
        for i in range(self.gdep):
            h = self.alpha * x + (1 - self.alpha) * self.nconv(h, a)
        ho = self.mlp(h)
        return ho


class mixprop(nn.Module):
    def __init__(self, c_in, c_out, gdep, dropout, alpha):
        super(mixprop, self).__init__()
        self.nconv = nconv()
        self.mlp = linear((gdep + 1) * c_in, c_out)
        self.gdep = gdep
        self.dropout = dropout
        self.alpha = alpha

    def forward(self, x, adj):
        if type(adj) != 'numpy.ndarray':
            adj = adj + torch.eye(adj.size(0)).to(x.device)
        else:
            adj = torch.from_numpy(adj) + torch.eye(adj.shape[0]).to(x.device)

        d = adj.sum(1)
        h = x
        out = [h]
        a = adj / d.view(-1, 1)

        for i in range(self.gdep):
            h = self.alpha * x + (1 - self.alpha) * self.nconv(h, a)
            out.append(h)
        ho = torch.cat(out, dim=1)
        ho = ho.float()
        ho = self.mlp(ho)
        return ho


class dy_mixprop(nn.Module):
    def __init__(self, c_in, c_out, gdep, dropout, alpha):
        super(dy_mixprop, self).__init__()
        self.nconv = dy_nconv()
        self.mlp1 = linear((gdep + 1) * c_in, c_out)
        self.mlp2 = linear((gdep + 1) * c_in, c_out)

        self.gdep = gdep
        self.dropout = dropout
        self.alpha = alpha
        self.lin1 = linear(c_in, c_in)
        self.lin2 = linear(c_in, c_in)

    def forward(self, x):
        # adj = adj + torch.eye(adj.size(0)).to(x.device)
        # d = adj.sum(1)
        x1 = torch.tanh(self.lin1(x))
        x2 = torch.tanh(self.lin2(x))
        adj = self.nconv(x1.transpose(2, 1), x2)
        adj0 = torch.softmax(adj, dim=2)
        adj1 = torch.softmax(adj.transpose(2, 1), dim=2)

        h = x
        out = [h]
        for i in range(self.gdep):
            h = self.alpha * x + (1 - self.alpha) * self.nconv(h, adj0)
            out.append(h)
        ho = torch.cat(out, dim=1)
        ho1 = self.mlp1(ho)

        h = x
        out = [h]
        for i in range(self.gdep):
            h = self.alpha * x + (1 - self.alpha) * self.nconv(h, adj1)
            out.append(h)
        ho = torch.cat(out, dim=1)
        ho2 = self.mlp2(ho)

        return ho1 + ho2


class dilated_1D(nn.Module):
    def __init__(self, cin, cout, dilation_factor=2):
        super(dilated_1D, self).__init__()
        self.tconv = nn.ModuleList()
        self.kernel_set = [2, 3, 6, 7]
        self.tconv = nn.Conv2d(cin, cout, (1, 7), dilation=(1, dilation_factor))

    def forward(self, input):
        x = self.tconv(input)
        return x


class dilated_inception(nn.Module):
    def __init__(self, cin, cout, dilation_factor=2):
        super(dilated_inception, self).__init__()
        self.tconv = nn.ModuleList()
        self.kernel_set = [2, 3, 6, 7]
        cout = int(cout / len(self.kernel_set))
        for kern in self.kernel_set:
            self.tconv.append(nn.Conv2d(cin, cout, (1, kern), dilation=(1, dilation_factor)))

    def forward(self, input):
        x = []
        for i in range(len(self.kernel_set)):
            x.append(self.tconv[i](input))
        for i in range(len(self.kernel_set)):
            x[i] = x[i][..., -x[-1].size(3):]
        x = torch.cat(x, dim=1)
        return x


class DilatedCausalConvolution(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, num_layers):
        super(DilatedCausalConvolution, self).__init__()
        self.layers = nn.ModuleList()
        self.num_layers = num_layers

        # Exponentially increasing dilation rates
        dilations = [2 ** i for i in range(num_layers)]

        # Creating dilated causal convolution layers
        for dilation in dilations:
            padding = (kernel_size - 1) * dilation  # Padding to maintain sequence length
            conv_layer = nn.Conv1d(input_channels, output_channels, kernel_size,
                                   padding=padding, dilation=dilation)
            self.layers.append(conv_layer)

    def forward(self, x):
        # Input x should have shape [batch_size, input_channels, sequence_length]

        # Applying each convolution layer
        for layer in self.layers:
            x = layer(x)
            # Removing extra padding to maintain causality
            x = x[:, :, :-((layer.kernel_size[0] - 1) * layer.dilation[0])]

        return x


class graph_constructor(nn.Module):
    def __init__(self, nnodes, k, dim, init_node_emb, device, original_wt=None, MTE=True, alpha=3, static_feat=None):
        super(graph_constructor, self).__init__()
        self.nnodes = nnodes
        if static_feat is not None:
            xd = static_feat.shape[1]
            self.lin1 = nn.Linear(xd, dim)
            self.lin2 = nn.Linear(xd, dim)
        else:
            self.emb1 = nn.Embedding(nnodes, init_node_emb)
            self.emb2 = nn.Embedding(nnodes, init_node_emb)
            if MTE:
                # The weight is initialized using MTE
                self.emb1.weight.data = original_wt
                self.emb2.weight.data = original_wt
                self.lin1 = nn.Linear(nnodes, dim)
                self.lin2 = nn.Linear(nnodes, dim)
            else:
                self.lin1 = nn.Linear(dim, dim)
                self.lin2 = nn.Linear(dim, dim)

        self.device = device
        self.k = k
        self.dim = dim
        self.alpha = alpha
        self.static_feat = static_feat

    def forward(self, idx):
        if self.static_feat is None:
            nodevec1 = self.emb1(idx)
            nodevec2 = self.emb2(idx)
        else:
            nodevec1 = self.static_feat[idx, :]
            nodevec2 = nodevec1

        nodevec1 = nodevec1.to(self.lin1.weight.dtype)
        nodevec2 = nodevec2.to(self.lin2.weight.dtype)

        nodevec1 = torch.tanh(self.alpha * self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha * self.lin2(nodevec2))

        a = torch.mm(nodevec1, nodevec2.transpose(1, 0)) - torch.mm(nodevec2, nodevec1.transpose(1, 0))
        temp = a.data.numpy()
        adj = F.relu(torch.tanh(self.alpha * a))
        mask = torch.zeros(idx.size(0), idx.size(0)).to(self.device)
        mask.fill_(float('0'))
        s1, t1 = (adj + torch.rand_like(adj) * 0.01).topk(self.k, 1)
        mask.scatter_(1, t1, s1.fill_(1))
        adj = adj * mask

        return adj

    def fullA(self, idx):
        if self.static_feat is None:
            nodevec1 = self.emb1(idx)
            nodevec2 = self.emb2(idx)
        else:
            nodevec1 = self.static_feat[idx, :]
            nodevec2 = nodevec1

        nodevec1 = torch.tanh(self.alpha * self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha * self.lin2(nodevec2))

        a = torch.mm(nodevec1, nodevec2.transpose(1, 0)) - torch.mm(nodevec2, nodevec1.transpose(1, 0))
        adj = F.relu(torch.tanh(self.alpha * a))
        return adj


class graph_global(nn.Module):
    def __init__(self, nnodes, k, dim, device, alpha=3, static_feat=None):
        super(graph_global, self).__init__()
        self.nnodes = nnodes
        self.A = nn.Parameter(torch.randn(nnodes, nnodes).to(device), requires_grad=True).to(device)

    def forward(self, idx):
        return F.relu(self.A)


class graph_undirected(nn.Module):
    def __init__(self, nnodes, k, dim, device, alpha=3, static_feat=None):
        super(graph_undirected, self).__init__()
        self.nnodes = nnodes
        if static_feat is not None:
            xd = static_feat.shape[1]
            self.lin1 = nn.Linear(xd, dim)
        else:
            self.emb1 = nn.Embedding(nnodes, dim)
            self.lin1 = nn.Linear(dim, dim)

        self.device = device
        self.k = k
        self.dim = dim
        self.alpha = alpha
        self.static_feat = static_feat

    def forward(self, idx):
        if self.static_feat is None:
            nodevec1 = self.emb1(idx)
            nodevec2 = self.emb1(idx)
        else:
            nodevec1 = self.static_feat[idx, :]
            nodevec2 = nodevec1

        nodevec1 = torch.tanh(self.alpha * self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha * self.lin1(nodevec2))

        a = torch.mm(nodevec1, nodevec2.transpose(1, 0))
        adj = F.relu(torch.tanh(self.alpha * a))
        mask = torch.zeros(idx.size(0), idx.size(0)).to(self.device)
        mask.fill_(float('0'))
        s1, t1 = adj.topk(self.k, 1)
        mask.scatter_(1, t1, s1.fill_(1))
        adj = adj * mask
        return adj


class graph_directed(nn.Module):
    def __init__(self, nnodes, k, dim, device, alpha=3, static_feat=None):
        super(graph_directed, self).__init__()
        self.nnodes = nnodes
        if static_feat is not None:
            xd = static_feat.shape[1]
            self.lin1 = nn.Linear(xd, dim)
            self.lin2 = nn.Linear(xd, dim)
        else:
            self.emb1 = nn.Embedding(nnodes, dim)
            self.emb2 = nn.Embedding(nnodes, dim)
            self.lin1 = nn.Linear(dim, dim)
            self.lin2 = nn.Linear(dim, dim)

        self.device = device
        self.k = k
        self.dim = dim
        self.alpha = alpha
        self.static_feat = static_feat

    def forward(self, idx):
        if self.static_feat is None:
            nodevec1 = self.emb1(idx)
            nodevec2 = self.emb2(idx)
        else:
            nodevec1 = self.static_feat[idx, :]
            nodevec2 = nodevec1

        nodevec1 = torch.tanh(self.alpha * self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha * self.lin2(nodevec2))

        a = torch.mm(nodevec1, nodevec2.transpose(1, 0))
        adj = F.relu(torch.tanh(self.alpha * a))
        mask = torch.zeros(idx.size(0), idx.size(0)).to(self.device)
        mask.fill_(float('0'))
        s1, t1 = adj.topk(self.k, 1)
        mask.scatter_(1, t1, s1.fill_(1))
        adj = adj * mask
        return adj


class LayerNorm(nn.Module):
    __constants__ = ['normalized_shape', 'weight', 'bias', 'eps', 'elementwise_affine']

    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super(LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.Tensor(*normalized_shape))
            self.bias = nn.Parameter(torch.Tensor(*normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.elementwise_affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def forward(self, input, idx):
        if self.elementwise_affine:
            return F.layer_norm(input, tuple(input.shape[1:]), self.weight[:, idx, :], self.bias[:, idx, :], self.eps)
        else:
            return F.layer_norm(input, tuple(input.shape[1:]), self.weight, self.bias, self.eps)

    def extra_repr(self):
        return '{normalized_shape}, eps={eps}, ' \
               'elementwise_affine={elementwise_affine}'.format(**self.__dict__)


class transfer_entropy_adaptively(nn.Module):
    def __init__(self, nnodes, k, dim, device, alpha=3):
        super(transfer_entropy_adaptively, self).__init__()
        self.nnodes = nnodes
        self.emb1 = nn.Embedding(nnodes, dim)
        self.emb2 = nn.Embedding(nnodes, dim)
        self.lin1 = nn.Linear(dim, dim)
        self.lin2 = nn.Linear(dim, dim)
        self.device = device
        self.k = k
        self.dim = dim
        self.alpha = alpha

    def forward(self, idx):
        if self.static_feat is None:
            nodevec1 = self.emb1(idx)
            nodevec2 = self.emb2(idx)
        else:
            nodevec1 = self.static_feat[idx, :]
            nodevec2 = nodevec1

        nodevec1 = torch.tanh(self.alpha * self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha * self.lin2(nodevec2))

        a = torch.mm(nodevec1, nodevec2.transpose(1, 0)) - torch.mm(nodevec2, nodevec1.transpose(1, 0))
        adj = F.relu(torch.tanh(self.alpha * a))
        mask = torch.zeros(idx.size(0), idx.size(0)).to(self.device)
        mask.fill_(float('0'))
        s1, t1 = (adj + torch.rand_like(adj) * 0.01).topk(self.k, 1)
        mask.scatter_(1, t1, s1.fill_(1))
        adj = adj * mask
        return adj


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
