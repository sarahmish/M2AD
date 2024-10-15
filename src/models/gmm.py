#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from statistics import NormalDist

import numpy as np
from scipy.stats import gamma
from sklearn.mixture import GaussianMixture

LOGGER = logging.Logger(__file__)

def _get_sum(name, sensors):
    return sum([1 for sensor in sensors if name in sensor])

def _divide(x, y):
    return x / y if y else 0

def _get_default(sensors):
    return {sensor: 1 for sensor in sensors}

def _find_weights(sensors, prefix=None) -> dict:
    prefix = prefix or _get_default(sensors)

    pre_weights = {
        sensor_type: _divide(sensor_weight, _get_sum(sensor_type, sensors))
        for sensor_type, sensor_weight in prefix.items()
    }

    weights = [pre_weights[k] for sensor in sensors for k in pre_weights if k in sensor]
    
    return weights

def _compute_cdf(gmm, x):
    means = gmm.means_.flatten()
    sigma = np.sqrt(gmm.covariances_).flatten()
    weights = gmm.weights_.flatten()
    cdf = 0
    for i in range(len(means)):
        cdf += weights[i]*NormalDist(mu=means[i], sigma=sigma[i]).cdf(x)
    
    return cdf

def _combine_pval(cdf, mode='train'):
    p_val = 2 * np.array(list(map(np.min, zip(1-cdf, cdf))))

    p_val[p_val < 1e-16] = 1e-16

    fisher_pval = - 2*np.log(p_val)
    
    return fisher_pval, p_val

class GMM:
    """Gaussian Mixture Model."""

    def _parse_components(self, n_components, sensors, default=1):
        if sensors is None: 
            if isinstance(n_components, dict):
                raise ValueError("Unknown list of sensors but specified in components.")
            
            elif isinstance(n_components, int):
                return n_components
        
            return default
            
        if isinstance(n_components, dict):
            n_components = {**n_components,
                **{k: default for k in sensors if k not in n_components}
            }

        elif isinstance(n_components, int):
            n_components = dict(zip(sensors, [n_components] * len(sensors)))

        return n_components
    

    def __init__(self, sensors, n_components=1, covariance_type='spherical', weights=None):
        self.sensors = sensors
        self.n_components = self._parse_components(n_components, sensors)
        self.covariance_type = covariance_type

        self.gmm = [None] * len(self.sensors)
        self.compute_cdf = np.vectorize(_compute_cdf)

        self.g_scale = None
        self.g_shape = None

        self.weights = weights or _find_weights(self.sensors)


    def fit(self, X):
        combined = 0
        num_sensors = X.shape[1]
        assert num_sensors == len(self.sensors)
        for i, sensor in enumerate(self.sensors):
            x = X[:, i].reshape(-1, 1)
            gmm = GaussianMixture(n_components=self.n_components[sensor],
                                  covariance_type=self.covariance_type)
            
            gmm.fit(x)
            self.gmm[i] = gmm

            cdf = self.compute_cdf(gmm, x.flatten())
            fisher, p_val = _combine_pval(cdf)

            combined += self.weights[i] * fisher

        if np.var(combined) > 0:
            self.g_scale = np.var(combined) / np.mean(combined)
            self.g_shape = np.mean(combined)**2 / np.var(combined)

        else:
            LOGGER.warning(f'No variance found between p-values ({np.var(combined)}).')
            self.g_scale = 1
            self.g_shape = 0


    def p_values(self, X):
        combined = 0
        p_val_sensors = np.zeros_like(X)
        fisher_values = np.zeros_like(X)
        for i, sensor in enumerate(self.sensors):
            y = X[:, i]
            gmm = self.gmm[i]
            cdf = self.compute_cdf(gmm, y)
            
            fisher, p_val = _combine_pval(cdf)
            combined += self.weights[i] * fisher

            p_val_sensors[:, i] = p_val
            fisher_values[:, i] = fisher

        gamma_p_val = 1 - gamma.cdf(combined, a=self.g_shape, scale=self.g_scale)

        return gamma_p_val, p_val_sensors, combined, fisher_values