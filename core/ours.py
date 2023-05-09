import os
import numpy as np
import pandas as pd
import copy
from scipy.optimize import brentq
from scipy.special import softmax
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.api import ExponentialSmoothing
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
from .conformal import standard_weighted_quantile
from .utils import moving_average
from tqdm import tqdm
import pdb
import warnings

"""
    UTILITY FUNCTIONS
"""
def pinball_loss(data, theta, beta):
    return beta*(data-theta)*((data-theta)>0).astype(int) + (1-beta)*(theta-data)*((theta-data)>0).astype(int)

def pinball_grad(data, theta, beta): # setting beta = 0.9 makes the loss 0 at the 90% quantile
    return -beta*((data-theta)>=0).astype(int) + (1-beta)*((theta-data)>0).astype(int)

"""
    BASELINES
"""

def trailing_window(
    scores,
    alpha,
    lr, # Dummy argument
    weight_length,
    *args,
    **kwargs
):
    T_test = scores.shape[0]
    qs = np.zeros((T_test,))
    for t in tqdm(range(T_test)):
        if min(weight_length, t) < np.ceil(1/alpha):
            qs[t] = np.infty
        else:
            qs[t] = np.quantile(scores[max(t-weight_length,0):t], 1-alpha, method='higher')
    results = {"method": "trail", "q" : qs}
    return results

def aci(
    scores,
    alpha,
    lr,
    window_length,
    T_burnin,
    *args,
    **kwargs
):
    T_test = scores.shape[0]
    alphat = alpha
    qs = np.zeros((T_test,))
    grads = np.zeros((T_test,))
    alphas = np.ones((T_test,)) * alpha
    for t in tqdm(range(T_test)):
        if t > T_burnin:
            # Setup: current gradient
            if alphat <= 1/(t+1):
                qs[t] = np.infty
            else:
                qs[t] = np.quantile(scores[max(t-window_length,0):t], 1-np.clip(alphat, 0, 1), method='higher')
            covered = qs[t] >= scores[t]
            grads[t] = -alpha if covered else 1-alpha
            alphat = alphat - lr*grads[t]

            if t < T_test - 1:
                alphas[t+1] = alphat
        else:
            if t > np.ceil(1/alpha):
                qs[t] = np.quantile(scores[:t], 1-alpha)
            else:
                qs[t] = np.infty
    results = { "method": "aci", "q" : qs, "gradient" : grads, "alpha" : alphas}
    return results

"""
    New methods
"""

def online_quantile(
    scores,
    alpha,
    lr,
    *args,
    **kwargs
):
    T_test = scores.shape[0]
    grads = np.zeros((T_test,))
    qs = np.zeros((T_test,))
    for t in tqdm(range(T_test)):
        # First, calculate the quantile in the current timestep
        covered = qs[t] >= scores[t]
        grads[t] = alpha if covered else -(1-alpha)
        # Gradient descent step
        if t < T_test - 1:
            qs[t+1] = qs[t] - lr * grads[t]
    results = {"method": "quantile", "q" : qs, "grads": grads}
    return results

def mytan(x):
    if x > np.pi/2:
        return np.infty
    elif x < -np.pi/2:
        return -np.infty
    else:
        return np.tan(x)

def barrier_fn(x, t, c1, c2, T_burnin):
    if t >= T_burnin:
        return mytan((x * np.log(t-T_burnin+1))/(c1*(t-T_burnin+100)))/c2
    else:
        return 0

def pi(
    scores,
    alpha,
    lr,
    c1,
    c2,
    T_burnin,
    *args,
    **kwargs
):
    T_test = scores.shape[0]
    grads = np.zeros((T_test,))
    qs = np.zeros((T_test,))
    shats = np.zeros((T_test,))
    sum_errors = 0
    for t in tqdm(range(T_test)):
        # First, calculate the quantile in the current timestep
        covered = qs[t] >= scores[t]
        if t > T_burnin:
            sum_errors += 1-covered
        grads[t] = alpha if covered else -(1-alpha)
        # PI step
        P = - lr * grads[t]
        if t > T_burnin:
            I = sum_errors - (t-T_burnin)*alpha
        else:
            I = 0
        if t < T_test - 1:
            shats[t+1] = shats[t] + P
            qs[t+1] = shats[t+1] + barrier_fn(I, t, c1, c2, T_burnin)
            #print("t: ", t, "shat: ", shats[t+1], "qhat: ", qs[t+1])
    results = {"method": "pi", "q" : qs, "grads": grads}
    return results

def pid(
    scores,
    alpha,
    lr,
    c1,
    c2,
    Kd,
    T_burnin,
    *args,
    **kwargs
):
    T_test = scores.shape[0]
    grads = np.zeros((T_test,))
    qs = np.zeros((T_test,))
    shats = np.zeros((T_test,))
    sum_errors = 0
    for t in tqdm(range(T_test)):
        # First, calculate the quantile in the current timestep
        covered = qs[t] >= scores[t]
        if t > T_burnin:
            sum_errors += 1-covered
        grads[t] = alpha if covered else -(1-alpha)
        # PI step
        P = - lr * grads[t]
        if t > T_burnin:
            I = sum_errors - (t-T_burnin)*alpha
            D = grads[t] - grads[t-1]
        else:
            I = 0
            D = 0
        if t < T_test - 1:
            shats[t+1] = shats[t] + P + Kd * D
            qs[t+1] = shats[t+1] + barrier_fn(I, t, c1, c2, T_burnin)
    results = {"method": "pid", "q" : qs, "grads": grads}
    return results

def pid_ets(
    scores,
    alpha,
    lr,
    data,
    T_burnin,
    c1,
    c2,
    steps_ahead,
    *args,
    **kwargs
):
    T_test = scores.shape[0]
    grads = np.zeros((T_test,))
    qs = np.zeros((T_test,))
    forecasts = np.zeros((T_test,))
    adj = np.zeros((T_test,))
    if data is None: # Data is synthetic
        data = pd.DataFrame({'timestamp': np.arange(scores.shape[0]), 'item_id': 'y', 'target': scores})
    if data['timestamp'].dtype == str:
        data['timestamp'] = pd.to_datetime(data['timestamp'])
    uq_timestamps = data['timestamp'].unique()
    data = data.set_index('timestamp')
    data = data[data.item_id == 'y']
    data.drop('item_id', axis=1, inplace=True)
    data['target'] = scores # KLUGE: We are forecasting scores now
    data = data.astype({'target': 'float'})
    seasonal_period = kwargs.get('seasonal_period')
    train_model = True
    try:
        os.makedirs('./.cache/pid_ets/', exist_ok=True)
        forecasts = np.load('./.cache/pid_ets/' + kwargs.get('config_name') + '.npy')
        train_model = False
    except:
        pass
    sum_errors = 0
    I = 0
    for t in tqdm(range(T_test)):
        if t > T_burnin:
            curr_steps_ahead = min(t+steps_ahead, T_test) - t
            curr_dates = uq_timestamps[t:t+curr_steps_ahead]
            initial_level = 0
            initial_trend = 0
            initial_seasonal = None if seasonal_period is None else 0
            if train_model and (t - T_burnin - 1) % steps_ahead == 0:
                # Use the statsmodels seasonal exponential smoothing to forecast the next steps_ahead quantiles
                curr_data = data[data.index < curr_dates[0]]
                model = ExponentialSmoothing(
                    curr_data,
                    seasonal_periods=seasonal_period,
                    trend='add',
                    seasonal=None if seasonal_period is None else 'add',
                    initial_level = initial_level,
                    initial_trend = initial_trend,
                    initial_seasonal = initial_seasonal,
                    initialization_method="known",
                ).fit()
                initial_level = model.level
                initial_trend = model.trend
                initial_seasonal = None if seasonal_period is None else model.season
                simulations = model.simulate(curr_steps_ahead, repetitions=100).to_numpy()
                forecasts[t:t+curr_steps_ahead] = np.quantile(simulations, 1-alpha, axis=1)
            qs[t] = forecasts[t] + adj[t] + barrier_fn(I, t, c1, c2, T_burnin)
            covered = qs[t] >= scores[t]
            sum_errors += 1-covered
            I = sum_errors - (t-T_burnin)*alpha
            grads[t] = alpha if covered else -(1-alpha)
            # Gradient descent step
            if t < T_test - 1:
                adj[t+1] = adj[t] - lr * grads[t]
        else:
            qs[t+1] = scores[t]
    results = {"method": "pid+ets", "q" : qs, "grads": grads}
    if train_model:
        np.save('./.cache/pid_ets/' + kwargs.get('dataset_name') + '.npy', forecasts)
    return results

def pid_gluon(
    scores,
    alpha,
    lr,
    data,
    T_burnin,
    c1,
    c2,
    steps_ahead,
    *args,
    **kwargs
):
    T_test = scores.shape[0]
    seasonal_period = kwargs.get('seasonal_period')
    grads = np.zeros((T_test,))
    qs = np.zeros((T_test,))
    forecasts = np.zeros((T_test,))
    adj = np.zeros((T_test,))
    if data is None: # Data is synthetic
        data = pd.DataFrame({'timestamp': np.arange(scores.shape[0]), 'item_id': 'y', 'target': scores})
    else:
        score_df = pd.DataFrame({'timestamp': data[data.item_id == 'y'].timestamp, 'item_id': np.array(['score']*data[data.item_id == 'y'].shape[0]), 'target': scores}, columns=['timestamp', 'item_id', 'target'])
        data = data[data.item_id != 'forecast']
        data = pd.concat([data, score_df])
        data.timestamp = pd.to_datetime(data.timestamp)
    data = data.fillna(0)
    data = data.astype({'target': 'float'})
    ignore_time_index = data.timestamp[:20].diff().min() != data.timestamp[:20].diff().max() # AutoGluon can't handle irregularly spaced data
    train_model = True
    try:
        os.makedirs('./.cache/pid_gluon/', exist_ok=True)
        forecasts = np.load('./.cache/pid_gluon/' + kwargs.get('dataset_name') + '.npy')
        train_model = False
    except:
        pass
    sum_errors = 0
    I = 0
    for t in tqdm(range(T_test)):
        if t > T_burnin:
            curr_steps_ahead = min(t+steps_ahead, T_test) - t
            curr_dates = data.timestamp.unique()[t:t+curr_steps_ahead]
            if train_model and (t - T_burnin - 1) % steps_ahead == 0:
                # Use the AutoGluon library to forecast the next steps_ahead quantiles
                curr_data = data[data.timestamp < curr_dates[0]]
                train_data = TimeSeriesDataFrame.from_data_frame(
                    curr_data,
                    id_column="item_id",
                    timestamp_column="timestamp"
                )
                predictor = TimeSeriesPredictor(
                    prediction_length=curr_steps_ahead,
                    path=None,
                    target="target",
                    eval_metric="sMAPE",
                    verbosity=0,
                    ignore_time_index=ignore_time_index
                )
                predictor.fit(
                    train_data,
                    presets="fast_training",
                    hyperparameters={
                        "ETS": {"seasonal_period": seasonal_period},
                    }
                )
                if ignore_time_index:
                    forecasts[t:t+curr_steps_ahead] = predictor.predict(curr_data, random_seed=None)[str(1-alpha)]['score'].to_numpy()
                else:
                    forecasts[t:t+curr_steps_ahead] = np.array([predictor.predict(curr_data, random_seed=None)[str(1-alpha)]['score', date] for date in curr_dates])
            qs[t] = forecasts[t] + adj[t] + barrier_fn(I, t, c1, c2, T_burnin)
            covered = qs[t] >= scores[t]
            sum_errors += 1-covered
            I = sum_errors - (t-T_burnin)*alpha
            grads[t] = alpha if covered else -(1-alpha)
            # Gradient descent step
            if t < T_test - 1:
                adj[t+1] = adj[t] - lr * grads[t]
        else:
            qs[t+1] = scores[t]
    results = {"method": "pid+gluon", "q" : qs, "grads": grads}
    if train_model:
        np.save('./.cache/pid_gluon/' + kwargs.get('dataset_name') + '.npy', forecasts)
    return results
