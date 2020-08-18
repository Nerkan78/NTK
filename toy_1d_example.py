


#@title ##### (Imports)
from importlib import reload
import numpy as usual_np
import jax.numpy as np
from jax import random
from jax import vmap

import functools

# from bayesian_ntk.utils import get_toy_data
import bayesian_ntk
from bayesian_ntk.models import homoscedastic_model
from bayesian_ntk.train import train_model
from bayesian_ntk.predict import Gaussian
from bayesian_ntk import predict, config, train_utils

from IPython.display import set_matplotlib_formats
set_matplotlib_formats('pdf', 'svg')
import matplotlib
import seaborn as sns
sns.set(font_scale=1.3)
sns.set_style("darkgrid", {"axes.facecolor": ".95"})
import matplotlib.pyplot as plt


from sklearn import preprocessing
from sklearn.model_selection import train_test_split

#@title ##### (GET DATA)
from jax import random
import math
from collections import namedtuple
import jax
print(jax.devices())
# Data = namedtuple(
#     'Data',
#     ['inputs', 'targets']
# )
#
# ''' Get data'''
# data = usual_np.loadtxt('dataset/data.txt')
#
# jax_x_data = np.array(data)[:, :-2]
# jax_y_data = np.array(data)[:, -1].reshape(-1,1)
#
# jax_x_data = preprocessing.StandardScaler().fit_transform(jax_x_data)
# X_train, X_test, y_train, y_test = train_test_split( jax_x_data, jax_y_data, test_size=0.6, random_state=42)
#
# train_data = Data( inputs = np.array(X_train), targets = y_train)
# test_data = Data( inputs = np.array(X_test), targets = y_test)
#
# from jax import device_put
# train_data = device_put(train_data)
# test_data = device_put(test_data)
#
# ''' Some utils'''
# model_config = config.get_model_config('default')
# _, _, kernel_fn = homoscedastic_model(parameterization = 'ntk', **model_config)
# key = random.PRNGKey(10);
#
#
# """## Making predictions"""
# analytic_ntkgp_moments, analytic_nngp_moments = predict.gp_inference(
#     kernel_fn = kernel_fn,
#     x_train = train_data.inputs,
#     y_train = train_data.targets,
#     x_test = test_data.inputs,
#     get = ('ntk', 'nngp'),
#     diag_reg = config.NOISE_SCALE**2,
#     compute_cov = True
# )
#
# predictions = {
#     'NTKGP analytic': analytic_ntkgp_moments,
#     'NNGP analytic': analytic_nngp_moments
# }
#
#
# ''' Train'''
#
# train_config = config.get_train_config('default')
# ensemble_key = random.split(key, config.ENSEMBLE_SIZE)
# train_baselearner = lambda key, train_method: train_model(key, train_method, train_data, test_data, parameterization = 'standard', **train_config)
# train_ensemble = lambda train_method: vmap(train_baselearner, (0, None))(ensemble_key, train_method)
#
# ensemble_methods_list = ['Deep ensemble', 'RP-param', 'NTKGP-param']
#
# # this may take a few minutes
# for method_idx, method in enumerate(ensemble_methods_list):
#     method_input_str = train_utils.method_input_dct[method]
#     print(f"Training ensemble method {method_idx+1}/{len(ensemble_methods_list)}: {method}")
#     baselearners_test_pred = train_ensemble(method_input_str)
#
#     # print(f"{method} : shape is {baselearners_test_pred.shape} \n pred is {baselearners_test_pred} ")
#     ensemble_mean = np.mean(baselearners_test_pred, axis = 0) #.reshape(-1,)
#
#     # ensemble_var = []
#     # for idx in range(baselearners_test_pred.shape[1]):
#     #     cov_matrix = usual_np.cov(baselearners_test_pred[:, idx, :].T)
#     #     ensemble_var.append(cov_matrix)
#
#     ensemble_var = np.var(baselearners_test_pred, axis = 0, ddof = 1) #.reshape(-1,)
#     # ensemble_std = np.sqrt(ensemble_var + config.NOISE_SCALE ** 2)
#
#     predictions.update(
#         {
#             method: Gaussian(ensemble_mean, ensemble_var)
#         }
#     )
#
# def estimate_log_likelihood(targets, means, stds ):
#     log_likelihood = 0
#     for target, mean, std in zip(targets, means, stds ):
#         log_likelihood -= np.log(np.sqrt(2 * np.pi * std)) + (target-mean)**2 / ( 2 * std)
#         # log_likelihood -= np.log(2 * np.pi * np.sqrt(np.linalg.det(std))) + (target - mean).T @np.linalg.inv(std)@(target - mean) / 2
#     return log_likelihood
#
# for method in predictions:
#     print(f"{method}: loglikelihood is {estimate_log_likelihood(test_data.targets, predictions[method].mean, predictions[method].standard_deviation )}")
#
