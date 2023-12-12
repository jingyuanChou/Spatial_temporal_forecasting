import pandas as pd
import argparse
import torch.nn as nn
from net import MTE_STJGC
from util import *
from trainer import Optim

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cpu', help='')
parser.add_argument('--data', type=str, default='data/METR-LA', help='data path')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--kernel_size', type=int, default=2)
parser.add_argument('--horizon', type=int, default=4)
parser.add_argument('--save', type=str, default='model/model.pt',
                    help='path to save the final model')
parser.add_argument('--num_nodes', type=int, default=49, help='number of nodes/variables')
parser.add_argument('--dropout', type=float, default=0.2, help='dropout rate')
parser.add_argument('--node_dim', type=int, default=16, help='dim of nodes')
parser.add_argument('--optim', type=str, default='adam')
parser.add_argument('--in_dim', type=int, default=1, help='inputs dimension')
parser.add_argument('--seq_in_len', type=int, default=12, help='input sequence length')
parser.add_argument('--seq_out_len', type=int, default=4, help='output sequence length')
parser.add_argument('--layers', type=int, default=4, help='number of layers')
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay rate')
parser.add_argument('--clip', type=int, default=5, help='clip')
parser.add_argument('--dilation_rates', type=list, default=[1, 2, 4, 4], help='dilation of each STJGC layer')
parser.add_argument('--L1Loss', type=bool, default=True)
parser.add_argument('--epochs', type=int, default=100, help='')
parser.add_argument('--seed', type=int, default=101, help='random seed')
parser.add_argument('--expid', type=int, default=1, help='experiment id')
parser.add_argument('--alpha', type=float, default=0.0, help='adj alpha')
parser.add_argument('--normalize', type=int, default=0)
parser.add_argument('--multivariate_TE_enhanced', type=bool, default=True, help='Multivariate Transfer Entropy')

args = parser.parse_args()
torch.set_num_threads(3)


def evaluate(data, X, Y, model, hosp, batch_size, max_ls, min_ls, stage, criterion):
    model.eval()
    total_loss = 0
    idx_count = 0
    MTE_index = data.validation_index_MTE
    for X, Y in data.get_batches(X, Y, batch_size):
        mte_index_for_valid = MTE_index[idx_count]
        idx_count = idx_count + 1
        X = torch.unsqueeze(X, dim=1)
        X = X.transpose(1, 2)
        with torch.no_grad():
            output = model(X, hosp, mte_index_for_valid).squeeze()
        # X_train_original = X_train_normalized * (max_val - min_val) + min_val
        original_Y = Y * (max_ls[stage] - min_ls[stage]) + min_ls[stage]
        output = torch.transpose(output, 0, 1) * (max_ls[stage] - min_ls[stage]) + min_ls[stage]
        loss = mape_loss(output,original_Y)
        total_loss += loss.item()
    return total_loss / idx_count




def train(data, X, Y, model, hosp, optim, batch_size, max_ls, min_ls, stage, criterion):
    model.train()
    total_loss = 0
    MTE_index = data.training_index_MTE
    idx_count = 0
    for X, Y in data.get_batches(X, Y, batch_size):
        mte_index_for_training = MTE_index[idx_count]
        idx_count = idx_count + 1
        model.zero_grad()
        X = torch.unsqueeze(X, dim=1)
        X = X.transpose(1, 2)

        output = model(X, hosp, mte_index_for_training).squeeze()
        original_Y = Y * (max_ls[stage] - min_ls[stage]) + min_ls[stage]
        output = torch.transpose(output,0,1) * (max_ls[stage] - min_ls[stage]) + min_ls[stage]
        loss = mape_loss(output,original_Y)
        if torch.isnan(loss).any():
            print("Loss is NaN")
        # print('the loss for {}-th batch is '.format(idx_count)+ str(loss.item()))
        total_loss += loss.item()
        loss.backward()
        optim.step()
    return total_loss/idx_count


def main(exp_num,name):

    hospitalization = pd.read_csv('hosp_weekly_filt_case_data.csv')
    hospitalization = hospitalization.reset_index()
    hospitalization['state'] = hospitalization['state'].astype(int)
    neighbouring_states = pd.read_csv('neighbouring_states.csv')
    fips_states = pd.read_csv('state_hhs_map.csv', header=None).iloc[:(-3), :]
    fips_2_state = fips_states.set_index(0)[2].to_dict()
    hospitalization['state'] = hospitalization['state'].map(fips_2_state)
    state_2_index = hospitalization.set_index('state')['index'].to_dict()
    neighbouring_states['StateCode'] = neighbouring_states['StateCode'].map(state_2_index)
    neighbouring_states['NeighborStateCode'] = neighbouring_states['NeighborStateCode'].map(state_2_index)
    hospitalization = hospitalization.iloc[:, 30:]  # remove all 0 datapoints
    hospitalization = hospitalization.T.values
    available_data = hospitalization




    if name == 'MTE_STJGC':

        for iter in range(exp_num):

            max_ls = dict()
            min_ls = dict()

            Data = Consecutive_dataloader(available_data, 0.7, 0.2, args.device, args.horizon, args.seq_in_len,
                                          args.horizon, args.normalize)

            # Step 1: Minimize/maximize over the first dimension (batches)
            min_val_1, _ = Data.train[0].min(dim=0)
            max_val_1, _ = Data.train[0].max(dim=0)
            # Step 2: Minimize/maximize over the second dimension (timestamps)
            min_val = min_val_1.min(dim=0)[0]
            max_val = max_val_1.max(dim=0)[0]
            # Normalize X_train
            Data.train[0] = (Data.train[0] - min_val) / (max_val - min_val)
            # Normalize Y_train
            Data.train[1] = (Data.train[1] - min_val) / (max_val - min_val)
            max_ls['train'] = max_val
            min_ls['train'] = min_val

            # Step 1: Minimize/maximize over the first dimension (batches)
            min_val_1, _ = Data.valid[0].min(dim=0)
            max_val_1, _ = Data.valid[0].max(dim=0)
            # Step 2: Minimize/maximize over the second dimension (timestamps)
            min_val = min_val_1.min(dim=0)[0]
            max_val = max_val_1.max(dim=0)[0]
            # Normalize X_val
            Data.valid[0] = (Data.valid[0] - min_val) / (max_val - min_val)
            # Normalize Y_val
            Data.valid[1] = (Data.valid[1] - min_val) / (max_val - min_val)
            max_ls['valid'] = max_val
            min_ls['valid'] = min_val

            # Step 1: Minimize/maximize over the first dimension (batches)
            min_val_1, _ = Data.testset[0].min(dim=0)
            max_val_1, _ = Data.testset[0].max(dim=0)
            # Step 2: Minimize/maximize over the second dimension (timestamps)
            min_val = min_val_1.min(dim=0)[0]
            max_val = max_val_1.max(dim=0)[0]
            # Normalize X_test
            Data.testset[0] = (Data.testset[0] - min_val) / (max_val - min_val)
            # Normalize Y_test
            Data.testset[1] = (Data.testset[1] - min_val) / (max_val - min_val)
            max_ls['test'] = max_val
            min_ls['test'] = min_val

            # Restore: X_train_original = X_train_normalized * (max_val - min_val) + min_val
            model = MTE_STJGC(args.batch_size, args.kernel_size, args.dilation_rates,
                          args.num_nodes, args.horizon, node_dim=args.node_dim, seq_length=args.seq_in_len, in_dim=args.in_dim,
                          layers=args.layers, alpha=args.alpha, MTE=args.multivariate_TE_enhanced)

            model = model.to(args.device)

            nParams = sum([p.nelement() for p in model.parameters()])
            print('Number of model parameters is', nParams, flush=True)

            if args.L1Loss:
                criterion = nn.L1Loss().to(args.device)
            else:
                criterion = nn.MSELoss().to(args.device)


            optim = Optim(
                model.parameters(), args.optim, args.lr, args.clip, lr_decay=args.weight_decay
            )

            print('begin training')
            train_losses = list()
            valid_losses = list()

            min_test_loss = float('inf')
            best_model_path = 'best_model.pth'
            for epoch in range(1, args.epochs + 1):
                train_loss = train(Data, Data.train[0], Data.train[1], model, available_data, optim,
                                   args.batch_size, max_ls, min_ls, 'train', criterion)
                val_loss = evaluate(Data, Data.valid[0], Data.valid[1], model, available_data,
                                   args.batch_size, max_ls, min_ls, 'valid', criterion)
                test_loss = evaluate(Data, Data.testset[0], Data.testset[1], model, available_data,
                                    args.batch_size, max_ls, min_ls, 'test', criterion)
                print(
                    '| end of epoch {:3d} | train_loss {:5.4f} | val_loss {:5.4f} | test_loss {:5.4f}'.format(
                        epoch, train_loss, val_loss, test_loss), flush=True)
                #                                       args.batch_size)
                # Save the model if the validation loss is the best we've seen so far.
                '''
                if min_test_loss < best_val:
                    with open(args.save, 'wb') as f:
                        torch.save(model, f)
                    best_val = val_loss
                '''
                train_losses.append(train_loss)
                # valid_losses.append(val_loss)
                '''
                if val_loss < min_val_loss:
                    min_val_loss = val_loss
                    # Save this model
                    torch.save(model.state_dict(), best_model_path)
                    best_adj_list = adj_list
                    best_epoch = epoch
                
                print(
                    '| end of epoch {:3d} | train_loss {:5.4f} | val_loss {:5.4f}'.format(
                        epoch, train_loss, val_loss), flush=True)
            '''
                '''
            plt.plot(train_losses, label = 'train_loss')
            plt.plot(valid_losses, label = 'validation_loss')
            plt.legend()
            plt.xlabel('epoch number')
            plt.ylabel('mae loss per sample')
            plt.show()
            '''
            best_model = MTE_STJGC(args.batch_size, args.kernel_size, args.dilation_rates,
                          args.num_nodes, args.horizon, node_dim=args.node_dim, seq_length=args.seq_in_len,
                                   in_dim=args.in_dim,
                          layers=args.layers, alpha=args.alpha, MTE=args.multivariate_TE_enhanced)
            best_model.load_state_dict(torch.load(best_model_path))

            # Predict in autoregressive way
        return

if __name__ == "__main__":
    # Selected baselines: MTGNN, STGCN, DCRNN, Graph WaveNet, STSGCN,ASTGCN, AGCRN, GMAN,STFGNN,GMSDR
    main(1,'MTE_STJGC')
