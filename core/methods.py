import os
import numpy as np
import pandas as pd
import copy
from statsmodels.tsa.api import ExponentialSmoothing
from tqdm import tqdm
import pdb
import warnings

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
    results = {"method": "Trail", "q" : qs}
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
    results = { "method": "ACI", "q" : qs, "gradient" : grads, "alpha" : alphas}
    return results

"""
    New methods
"""

def quantile(
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
    results = {"method": "Quantile", "q" : qs, "grads": grads}
    return results

def mytan(x):
    if x >= np.pi/2:
        return np.infty
    elif x <= -np.pi/2:
        return -np.infty
    else:
        return np.tan(x)

def saturation_fn_log(x, t, Csat, KI, T_burnin):
    if t >= T_burnin:
        return KI * mytan((x * np.log(t-T_burnin+1))/((Csat * t-T_burnin+100)))
    else:
        return 0

def saturation_fn_sqrt(x, t, Csat, KI, T_burnin):
    if t >= T_burnin:
        return KI * mytan((x * np.sqrt(t-T_burnin+1))/((Csat * t-T_burnin+100)))
    else:
        return 0

def quantile_integrator_log(
    scores,
    alpha,
    lr,
    Csat,
    KI,
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
            qs[t+1] = shats[t+1] + saturation_fn_log(I, t, Csat, KI, T_burnin)
    results = {"method": "Quantile+Integrator (log)", "q" : qs, "grads": grads}
    return results

def quantile_integrator_log_scorecaster(
    scores,
    alpha,
    lr,
    data,
    T_burnin,
    Csat,
    KI,
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
        os.makedirs('./.cache/scorecaster/', exist_ok=True)
        forecasts = np.load('./.cache/scorecaster/' + kwargs.get('config_name') + '.npy')
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
            qs[t] = forecasts[t] + adj[t] + saturation_fn_log(I, t, Csat, KI, T_burnin)
            covered = qs[t] >= scores[t]
            sum_errors += 1-covered
            I = sum_errors - (t-T_burnin)*alpha
            grads[t] = alpha if covered else -(1-alpha)
            # Gradient descent step
            if t < T_test - 1:
                adj[t+1] = adj[t] - lr * grads[t]
        else:
            qs[t+1] = scores[t]
    results = {"method": "Quantile+Integrator (log)+Scorecaster", "q" : qs, "grads": grads}
    if train_model:
        np.save('./.cache/scorecaster/' + kwargs.get('config_name') + '.npy', forecasts)
    return results