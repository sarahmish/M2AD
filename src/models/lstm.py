#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader

class Model(torch.nn.Module):
    def __init__(self, seq_len, in_channels, out_channels, lstm_units=80, n_layer=1, dropout=0.2):
        super(Model, self).__init__()
        self.seq_len = seq_len
        self.in_channels, self.out_channels = in_channels, out_channels
        self.lstm_units = lstm_units
        self.lstm1 = torch.nn.LSTM(
            input_size=self.in_channels,
            hidden_size=self.lstm_units,
            num_layers=n_layer,
            batch_first=True,
            dropout=dropout
        )
        self.lstm2 = torch.nn.LSTM(
            input_size=self.lstm_units,
            hidden_size=self.lstm_units,
            batch_first=True
        )
        self.dense = torch.nn.Linear(self.lstm_units, self.out_channels)

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.reshape((batch_size, self.seq_len, self.in_channels))
        x, (_, _) = self.lstm1(x)
        x, (hidden_n, _) = self.lstm2(x)
        hidden_n = hidden_n.reshape((batch_size, self.lstm_units))
        out = self.dense(hidden_n)
        return out
    

class EarlyStopping:
    def __init__(self, tolerance=5, min_delta=0):
        self.tolerance = tolerance
        self.min_delta = min_delta
        self.best_score = np.inf
        self.counter = 0
        self.early_stop = False

    def __call__(self, train_loss, validation_loss):
        if (validation_loss - train_loss) > self.min_delta or validation_loss >= self.best_score:
            self.counter +=1
            if self.counter >= self.tolerance:  
                self.early_stop = True

        else:
            self.counter = 0

        if validation_loss < self.best_score:
            self.best_score = validation_loss


class LSTM:
    """LSTM Forecasting Model."""
    def __init__(self, out_channels=None, lstm_units=80, n_layer=1, dropout=0.2, device='cuda', 
                 batch_size=32, lr=1e-3, verbose=True):
        self.out_channels = out_channels
        self.lstm_units = lstm_units
        self.n_layer = n_layer
        self.dropout = dropout
        self.device = device
        self.batch_size = batch_size
        self.lr = lr
        self.verbose = verbose

        # infer
        self.seq_len = None
        self.n_channels = None

    
    def fit(self, X, y, validation_split=0.2, epochs=35, tolerance=5, min_delta=10, checkpoint=50, path='checkpoint'):
        validation_size = int(len(X) * validation_split)
        train, train_y = torch.Tensor(X[:-validation_size]), torch.Tensor(y[:-validation_size])
        valid, valid_y = torch.Tensor(X[-validation_size:]), torch.Tensor(y[-validation_size:])

        # get shape
        _, self.seq_len, self.n_channels = X.shape
        if self.out_channels is None:
            self.out_channels = y.shape[1]

        train_loader = DataLoader(TensorDataset(train, train_y), batch_size=self.batch_size, shuffle=True)
        valid_loader = DataLoader(TensorDataset(valid, valid_y), batch_size=self.batch_size)
        
        self.model = Model(
            self.seq_len, 
            self.n_channels,
            self.out_channels,
            self.lstm_units, 
            self.n_layer, 
            self.dropout
        ).to(self.device)

        optimizer = Adam(self.model.parameters(), lr=self.lr)
        criterion = torch.nn.MSELoss(reduction='mean').to(self.device)
        self.history = dict(train=[], valid=[])

        early_stopping = EarlyStopping(tolerance=tolerance, min_delta=min_delta)

        for epoch in range(epochs):
            self.model.train()

            train_losses = []
            valid_losses = []

            for (x, y) in train_loader:
                optimizer.zero_grad()
                x = x.to(self.device)
                y = y.to(self.device)
                pred = self.model(x)
                loss = criterion(pred, y)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())
        
            self.model.eval()
            
            for i, (x, y) in enumerate(valid_loader):
                x = x.to(self.device)
                y = y.to(self.device)
                pred = self.model(x)
                loss = criterion(pred, y)
                valid_losses.append(loss.item())

            train_loss = np.mean(train_losses)
            valid_loss = np.mean(valid_losses)

            if self.verbose:
                print(f'Epoch {epoch+1}/{epochs}: train loss {train_loss: .4f} | val loss {valid_loss: .4f}.')

            self.history['train'].append(train_loss)
            self.history['valid'].append(valid_loss)

            if epoch % checkpoint == 0:
                os.makedirs(path, exist_ok=True)
                file = os.path.join(path, f'model-{epoch}.pth')
                torch.save(self.model.state_dict(), file)

            early_stopping(train_loss, valid_loss)
            if early_stopping.early_stop:
                if self.verbose:
                    print(f'Early stopping at epoch: {epoch + 1}')
                break


    def predict(self, X):
        test_loader = DataLoader(dataset=X, batch_size=1)

        output = list()
        self.model.eval()
        for x in test_loader:
            x = x.to(self.device)
            pred = self.model(x)
            pred = pred.squeeze().to('cpu').detach().numpy()
            output.append(pred)

        return np.array(output)