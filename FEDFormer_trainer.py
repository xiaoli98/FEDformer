import argparse
import numpy as np
import pandas as pd
import datetime
import dateutil.parser as time_parser
import os
import time
import warnings

import numpy as np
import lightning as L
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
import torch.amp
import torch.nn as nn
from torch.utils.data import Dataset 
from torch.utils.data import DataLoader 
from torch import optim
from models import FEDformer, Autoformer, Informer, Transformer
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
from utils.timefeatures import time_features

class FEDFormerForgingData(Dataset):
    def __init__(self, 
        root_path="../data/", 
        preprocessed_data="processed_forging_data.pkl",
        data_path="timeseries_1month.csv",
        test_data_path="timeseries.csv"):
        
        # forecasting task info
        self.seq_len = 24 * 4 * 4
        self.label_len = 24 * 4
        self.pred_len = 24 * 4
        
        if preprocessed_data:
            self.load(root_path+preprocessed_data)
        else:
            self.data = pd.read_csv(root_path + data_path)
            self.data_test = pd.read_csv(root_path + test_data_path)
            
            self.data.drop_duplicates(["Received Time"], inplace=True, ignore_index=True)
            self.data.drop(columns=["Unnamed: 0"])
            self.data_test.drop_duplicates(["Received Time"], inplace=True, ignore_index=True)
            self.data_test.drop(columns=["Unnamed: 0"])
            
            # self.data["Received Time"] = self.data["Received Time"].apply(time_parser.parse)
            self.data["Received Time"] = pd.to_datetime(self.data["Received Time"], format="ISO8601")
            self.data = self.data.set_index("Received Time")
            self.data = self.data.resample("100ms").ffill()
            self.data.dropna(inplace=True)
            self.data = self.data.reset_index()
            
            # self.data_test["Received Time"] = self.data_test["Received Time"].apply(time_parser.parse)
            self.data_test["Received Time"] = pd.to_datetime(self.data_test["Received Time"], format="ISO8601")
            self.data_test = self.data_test.set_index("Received Time")
            self.data_test = self.data_test.resample("100ms").ffill()
            self.data_test.dropna(inplace=True)
            self.data_test = self.data_test.reset_index()
            
            # time features encoded secondly
            self.data_stamp = time_features(pd.to_datetime(self.data["Received Time"].values), freq="s")
            self.data_stamp_test = time_features(pd.to_datetime(self.data_test["Received Time"].values), freq="s")
            self.data_stamp = self.data_stamp.transpose(1, 0)
            self.data_stamp_test = self.data_stamp_test.transpose(1, 0)
            
            self.filter = set()
            with open("~/EPICMAP/src/opcua/data_analysis/useless_counters.txt", "r") as f:
                to_filter = f.readline()
                while to_filter:
                    self.filter.add(to_filter.rstrip())
                    to_filter = f.readline()
            
            self.data = self.data.drop(columns=self.filter, errors="ignore")
            self.data_test = self.data_test.drop(columns=self.filter, errors="ignore")
            
            self.data = self.data.drop(columns=["Received Time"], errors="ignore")
            self.data_test = self.data_test.drop(columns=["Received Time"], errors="ignore")
            
            #scaling
            self.scaler = StandardScaler()
            self.data = self.scaler.fit_transform(self.data)
            self.data_test = self.scaler.transform(self.data_test)
            
            #aliases
            self.data_x = self.data
            self.data_y = self.data
                
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def save(self, path):
        to_save = {}
        to_save["data"] = self.data
        to_save["data_val"] = self.data_val
        to_save["data_stamp"] = self.data_stamp
        to_save["data_stamp_val"] = self.data_stamp_val
        to_save["filter"] = self.filter
        to_save["scaler"] = self.scaler
        with open(path, "wb") as f:
            pickle.dump(to_save, f, protocol=4)
    
    def load(self, path):
        with open(path, "rb") as f:
            to_load = pickle.load(f)
            self.data = to_load["data"]
            self.data_val = to_load["data_val"]
            self.data_stamp = to_load["data_stamp"]
            self.data_stamp_val = to_load["data_stamp_val"]
            self.filter = to_load["filter"]
            self.scaler = to_load["scaler"]
            self.data_x = self.data
            self.data_y = self.data


class LightningFEDFormer(L.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.model = FEDformer.Model(args)
        # self.model = torch.compile(self.model)
        self.criterion = nn.MSELoss()
        
        self.automatic_optimization = False
        self.save_hyperparameters(args)

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        opt.zero_grad()
        batch_x, batch_y, batch_x_mark, batch_y_mark = batch
        batch_x = batch_x.float()
        batch_y = batch_y.float()
        batch_x_mark = batch_x_mark.float()
        batch_y_mark = batch_y_mark.float()

        dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :], device=self.device).float()
        dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float()

        if self.args.use_amp:
            with torch.cuda.amp.autocast("cuda"):
                if self.args.output_attention:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        else:
            if self.args.output_attention:
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
            else:
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        f_dim = -1 if self.args.features == 'MS' else 0
        batch_y = batch_y[:, -self.args.pred_len:, f_dim:]
        loss = self.criterion(outputs, batch_y)
        # loss.backward()
        # opt.step()
        self.log_dict({"train_loss": loss}, prog_bar=True)
        return loss
        
    def validation_step(self, batch, batch_idx):
        batch_x, batch_y, batch_x_mark, batch_y_mark = batch
        batch_x = batch_x.float()
        batch_y = batch_y.float()
        batch_x_mark = batch_x_mark.float()
        batch_y_mark = batch_y_mark.float()

        dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
        dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float()
        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                if self.args.output_attention:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        else:
            if self.args.output_attention:
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
            else:
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        f_dim = -1 if self.args.features == 'MS' else 0
        batch_y = batch_y[:, -self.args.pred_len:, f_dim:]

        pred = outputs.detach().cpu()
        true = batch_y.detach().cpu()

        loss = self.criterion(pred, true)
        self.log_dict({"val_loss": loss}, prog_bar=True)

    def configure_optimizers(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

# region parser
parser = argparse.ArgumentParser(description='Autoformer & Transformer family for Time Series Forecasting')

# basic config
parser.add_argument('--is_training', type=int, default=1, help='status')
parser.add_argument('--task_id', type=str, default='test', help='task id')
parser.add_argument('--model', type=str, default='FEDformer',
                    help='model name, options: [FEDformer, Autoformer, Informer, Transformer]')

# supplementary config for FEDformer model
parser.add_argument('--version', type=str, default='Fourier',
                    help='for FEDformer, there are two versions to choose, options: [Fourier, Wavelets]')
parser.add_argument('--mode_select', type=str, default='random',
                    help='for FEDformer, there are two mode selection method, options: [random, low]')
parser.add_argument('--modes', type=int, default=64, help='modes to be selected random 64')
parser.add_argument('--L', type=int, default=3, help='ignore level')
parser.add_argument('--base', type=str, default='legendre', help='mwt base')
parser.add_argument('--cross_activation', type=str, default='tanh',
                    help='mwt cross atention activation function tanh or softmax')

# data loader
parser.add_argument('--data', type=str, default='ForgingData', help='dataset type')
parser.add_argument('--root_path', type=str, default='../../data/', help='root path of the data file')
parser.add_argument('--preprocesssed_path', type=str, default='processed_forging_data.pkl', help='pre processed data file')
parser.add_argument('--data_path', type=str, default='timeseries_1month.csv', help='data file')
parser.add_argument('--test_data_path', type=str, default='timeseries.csv', help='test data file')
parser.add_argument('--ratio', type=float, default=0.2, help='train val ratio')

parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, '
                            'S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='s',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, '
                            'b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--detail_freq', type=str, default='h', help='like freq, but use in predict')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

# forecasting task
parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
parser.add_argument('--label_len', type=int, default=96, help='start token length')
parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
# parser.add_argument('--cross_activation', type=str, default='tanh'

# model define
parser.add_argument('--enc_in', type=int, default=105, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=105, help='decoder input size')
parser.add_argument('--c_out', type=int, default=105, help='output size')
parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
parser.add_argument('--moving_avg', default=[12], help='window size of moving average')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--distil', action='store_false',
                    help='whether to use distilling in encoder, using this argument means not using distilling',
                    default=True)
parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

# optimization
parser.add_argument('--num_workers', type=int, default=32, help='data loader num workers')
parser.add_argument('--itr', type=int, default=3, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
parser.add_argument('--batch_size', type=int, default=128, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='mse', help='loss function')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

# GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1', help='device ids of multi gpus')

# randomness
parser.add_argument('--seed', type=int, default=42, help='random seed')

#endregion

args = parser.parse_args()

# Low-precision Matrix Multiplication
# Default used by PyTorch
# torch.set_float32_matmul_precision("highest")
# Faster, but less precise
# torch.set_float32_matmul_precision("high")
# Even faster, but also less precise
torch.set_float32_matmul_precision("medium")

fdata = FEDFormerForgingData(args.root_path, args.preprocesssed_path, args.data_path, args.test_data_path)

train_data, val_data = train_test_split(fdata, test_size=args.ratio, random_state=args.seed)
train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

model = LightningFEDFormer(args)

trainer = L.Trainer(
    devices=args.devices, 
    accelerator="gpu", 
    max_epochs=args.train_epochs,
    default_root_dir="./checkpoints/",
    )
trainer.fit(model, train_loader, val_loader)