#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import logging
import pickle
import operator
from itertools import compress

import numpy as np
import pandas as pd
from scipy.stats import gamma
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer

from .preprocessing import sliding_window_sequences
from .models.lstm import LSTM
from .models.gmm import GMM
from .errors import area_errors, point_errors

LOGGER = logging.Logger(__file__)


def _merge_sequences(sequences):
    if len(sequences) == 0:
        return np.array([])

    sorted_sequences = sorted(sequences, key=operator.itemgetter(0))
    new_sequences = [sorted_sequences[0]]
    score = [sorted_sequences[0][2]]

    for sequence in sorted_sequences[1:]:
        prev_sequence = new_sequences[-1]

        if sequence[0] <= prev_sequence[1] + 1:
            score.append(sequence[2])
            average = np.mean(score)
            new_sequences[-1] = (prev_sequence[0], max(prev_sequence[1], sequence[1]), average)
        else:
            score = [sequence[2]]
            new_sequences.append(sequence)

    return np.array(new_sequences)


class M2AD:
    def __init__(self, dataset, entity, sensors=None, covariates=None, time_column='time', 
                 feature_range=(0, 1), strategy='mean', error_name='point',
                 window_size=100, target_size=1, step_size=1, lstm_units=80, n_layer=2, 
                 dropout=0.2, device='cpu', batch_size=32, lr=1e-3, epochs=35, score_window=10, 
                 verbose=True, n_components=1, covariance_type='spherical', **kwargs):
        self.dataset = dataset
        self.entity = entity
        self.sensors = sensors
        self.covariates = covariates
        self.time_column = time_column

        self.scaler = MinMaxScaler(feature_range=feature_range)
        self.imputer = SimpleImputer(strategy=strategy)

        self.window_size = window_size
        self.target_size = target_size
        self.step_size = step_size

        self.model = LSTM(
            lstm_units=lstm_units,
            n_layer=n_layer,
            dropout=dropout,
            device=device,
            batch_size=batch_size,
            lr=lr,
            verbose=verbose
        )

        self.error_name = error_name
        self.gamma_thresh  = 0.001

        self.epochs = epochs
        self.score_window = score_window

        self.n_components = n_components
        self.covariance_type = covariance_type

        if self.error_name == 'point':
            self.one_sided = True
        elif self.error_name == 'area':
            self.one_sided = False
        else:
            raise ValueError(f"Unknown error function {self.error_name}.")

    def _get_data(self, df):
        data = df.copy()
        timestamp = data.pop(self.time_column)

        if self.covariates:
            X = data[self.sensors]
            cov = data[self.covariates]

        X = self.imputer.fit_transform(X)
        X = self.scaler.fit_transform(X)

        y = X.copy()

        X = np.concatenate([X, cov.values], axis=1) if self.covariates else X

        return X, y, cov, timestamp
    
    def _compute_error(self, y, pred):
        if self.error_name == 'point':
            return point_errors(y, pred, smooth=True)
        elif self.error_name == 'area':
            return area_errors(y, pred, smooth=True)
        
        raise ValueError(f"Unknown error function {self.error_name}.")

    def create_intervals(self, anomalies, index, score, anomaly_padding=50):
        intervals = list()
        length = len(anomalies)
        anomalies_index = list(compress(range(length), anomalies))
        for idx in anomalies_index:
            start = max(0, idx - anomaly_padding)
            end = min(idx + anomaly_padding + 1, length)
            value = np.mean(score[start: end])
            intervals.append([index[start], index[end], value])
        
        intervals = _merge_sequences(intervals)
        intervals = sorted(intervals, key=operator.itemgetter(0), reverse=True)

        anomalies = pd.DataFrame(intervals, columns=['start', 'end', 'score'])
        anomalies.insert(0, 'dataset', self.dataset)
        anomalies.insert(1, 'entity', self.entity)

        return anomalies

    def fit(self, df, validation_split=0.2, tolerance=10, min_delta=10):
        X, y, cov, timestamp = self._get_data(df)

        assert len(cov.columns) == len(self.covariates), f'check covariates {cov.columns}'
        assert y.shape[1] == len(self.sensors), f'check sensors {y.shape}'
        assert X.shape[1] == (len(self.covariates) + len(self.sensors)), f'check input {X.shape}'

        windows, targets, indices = sliding_window_sequences(
            X=X,
            y=y,
            index=timestamp,
            window_size=self.window_size,
            target_size=self.target_size,
            step_size=self.step_size
        )

        LOGGER.info(f'Training the model with {self.epochs} epochs.')
        self.model.fit(windows, targets, epochs=self.epochs, validation_split=validation_split, 
                       tolerance=tolerance, min_delta=min_delta)
        
        pred = self.model.predict(windows)
        pred = pred.reshape(targets.shape)
        
        errors = self._compute_error(targets, pred)

        self.gmm_model = GMM(sensors=self.sensors, 
                             n_components=self.n_components, 
                             covariance_type=self.covariance_type,
                             one_sided=self.one_sided)

        self.gmm_model.fit(errors)
        anomalyscore, pval, fisher, _ = self.gmm_model.p_values(errors)
        self.threshold = np.percentile(fisher, 99.5)

        self.train_mse = mean_squared_error(targets, pred)
        self.train_errors = errors
        self.train_targets = targets
        self.train_pred = pred
        self.train_pvals = pval
        self.train_anomalyscore = anomalyscore
                

    def detect(self, df, debug=False):
        X, y, _, timestamp = self._get_data(df)
        windows, targets, indices = sliding_window_sequences(
            X=X,
            y=y,
            index=timestamp,
            window_size=self.window_size,
            target_size=self.target_size,
            step_size=self.step_size
        )

        pred = self.model.predict(windows)
        pred = pred.reshape(targets.shape)
        errors = self._compute_error(targets, pred)

        LOGGER.info(f'Applying threshold on p-values.')
        anomalyscore, pval, fisher, fisher_values = self.gmm_model.p_values(errors)
        anomalies = anomalyscore < self.gamma_thresh
        formatted_anomalies = self.create_intervals(anomalies, indices, anomalyscore)

        if debug:
            visuals = {
                "anomalies": anomalies,
                "test_targets": targets,
                "test_pred": pred,
                "test_errors": errors,
                "test_pvals": pval,
                "test_anomaly_score": anomalyscore,
                "test_timestamps": indices,
                "train_targets": self.train_targets,
                "train_pred": self.train_pred,
                "train_errors": self.train_errors,
                "train_pvals": self.train_pvals,
                "train_anomaly_score": self.train_anomalyscore,
            }
            
            return formatted_anomalies, visuals

        return formatted_anomalies