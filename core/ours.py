import numpy as np
import copy
from scipy.optimize import brentq
from scipy.special import softmax
from statsmodels.tsa.arima.model import ARIMA
#from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
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
    results = {"method": "pid", "q" : qs, "grads": grads}
    return results

def arima_quantile(
    scores,
    alpha,
    lr,
    order,
    window_length,
    T_burnin,
    *args,
    **kwargs
):
    T_test = scores.shape[0]
    grads = np.zeros((T_test,))
    qs = np.zeros((T_test,))
    adj = np.zeros((T_test,))
    start_params = None
    for t in tqdm(range(T_test)):
        if t > T_burnin:
            # First, forecast the quantile in the current timestep
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ARIMAModel = ARIMA(scores[t-window_length:t], order=order, enforce_stationarity=False, enforce_invertibility=False).fit(start_params=start_params)
            qs[t] = ARIMAModel.get_forecast(t).conf_int(alpha=2*alpha)[0,1] + adj[t]
            start_params = ARIMAModel.params
            covered = qs[t] + adj[t] >= scores[t]
            grads[t] = alpha if covered else -(1-alpha)
            # Gradient descent step
            if t < T_test - 1:
                adj[t+1] = adj[t] - lr * grads[t]
        else:
            qs[t+1] = scores[t]
    results = {"method": "arima+quantile", "q" : qs, "grads": grads}
    return results

def pid_gluon(
    scores,
    datetimes,
    alpha,
    lr,
    order,
    window_length,
    T_burnin,
    c1,
    c2,
    *args,
    **kwargs
):
    T_test = scores.shape[0]
    grads = np.zeros((T_test,))
    qs = np.zeros((T_test,))
    adj = np.zeros((T_test,))
    for t in tqdm(range(T_test)):
        if t > T_burnin:
            # Use the AutoGluon library to forecast the next quantile
            pdb.set_trace()
            qs[t] = AutoGluonForecast(scores[t-window_length:t], datetimes[t-window_length:t], alpha=alpha, lr=lr, order=order, window_length=window_length, T_burnin=T_burnin)
            covered = qs[t] + adj[t] >= scores[t]
            grads[t] = alpha if covered else -(1-alpha)
            # Gradient descent step
            if t < T_test - 1:
                adj[t+1] = adj[t] - lr * grads[t]
        else:
            qs[t+1] = scores[t]
    results = {"method": "pid+gluon", "q" : qs, "grads": grads}
    return results
