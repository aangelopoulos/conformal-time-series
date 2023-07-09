import os
import numpy as np
import pandas as pd
import copy
from statsmodels.tsa.api import ExponentialSmoothing
from statsmodels.tsa.forecasting.theta import ThetaModel
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
    ahead,
    *args,
    **kwargs
):
    T_test = scores.shape[0]
    qs = np.zeros((T_test,))
    for t in tqdm(range(T_test)):
        t_pred = t - ahead + 1
        if min(weight_length, t_pred) < np.ceil(1/alpha):
            qs[t] = np.infty
        else:
            qs[t] = np.quantile(scores[max(t_pred-weight_length,0):t_pred], 1-alpha, method='higher')
    results = {"method": "Trail", "q" : qs}
    return results

def aci_clipped(
    scores,
    alpha,
    lr,
    window_length,
    T_burnin,
    ahead,
    *args,
    **kwargs
):
    T_test = scores.shape[0]
    alphat = alpha
    qs = np.zeros((T_test,))
    alphas = np.ones((T_test,)) * alpha
    covereds = np.zeros((T_test,))
    for t in tqdm(range(T_test)):
        t_pred = t - ahead + 1
        clip_value = scores[max(t_pred-window_length,0):t_pred].max() if t_pred > 0 else np.infty
        if t_pred > T_burnin:
            # Setup: current gradient
            if alphat <= 1/(t_pred+1):
                qs[t] = np.infty
            else:
                qs[t] = np.quantile(scores[max(t_pred-window_length,0):t_pred], 1-np.clip(alphat, 0, 1), method='higher')
            covereds[t] = qs[t] >= scores[t]
            grad = -alpha if covereds[t_pred] else 1-alpha
            alphat = alphat - lr*grad

            if t < T_test - 1:
                alphas[t+1] = alphat
        else:
            if t_pred > np.ceil(1/alpha):
                qs[t] = np.quantile(scores[:t_pred], 1-alpha)
            else:
                qs[t] = np.infty
        if qs[t] == np.infty:
            qs[t] = clip_value
    results = { "method": "ACI (clipped)", "q" : qs, "alpha" : alphas}
    return results


def aci(
    scores,
    alpha,
    lr,
    window_length,
    T_burnin,
    ahead,
    *args,
    **kwargs
):
    T_test = scores.shape[0]
    alphat = alpha
    qs = np.zeros((T_test,))
    alphas = np.ones((T_test,)) * alpha
    covereds = np.zeros((T_test,))
    for t in tqdm(range(T_test)):
        t_pred = t - ahead + 1
        if t_pred > T_burnin:
            # Setup: current gradient
            if alphat <= 1/(t_pred+1):
                qs[t] = np.infty
            else:
                qs[t] = np.quantile(scores[max(t_pred-window_length,0):t_pred], 1-np.clip(alphat, 0, 1), method='higher')
            covereds[t] = qs[t] >= scores[t]
            grad = -alpha if covereds[t_pred] else 1-alpha
            alphat = alphat - lr*grad

            if t < T_test - 1:
                alphas[t+1] = alphat
        else:
            if t_pred > np.ceil(1/alpha):
                qs[t] = np.quantile(scores[:t_pred], 1-alpha)
            else:
                qs[t] = np.infty
    results = { "method": "ACI", "q" : qs, "alpha" : alphas}
    return results

"""
    New methods
"""

def quantile(
    scores,
    alpha,
    lr,
    ahead,
    proportional_lr=True,
    *args,
    **kwargs
):
    T_burnin = kwargs['T_burnin']
    results = quantile_integrator_log(scores, alpha, lr, 1.0, 0, ahead, T_burnin, proportional_lr=proportional_lr)
    results['method'] = 'Quantile'
    return results

def mytan(x):
    if x >= np.pi/2:
        return np.infty
    elif x <= -np.pi/2:
        return -np.infty
    else:
        return np.tan(x)

def saturation_fn_log(x, t, Csat, KI):
    if KI == 0:
        return 0
    tan_out = mytan(x * np.log(t+1)/(Csat * (t+1)))
    out = KI * tan_out
    return  out

def saturation_fn_sqrt(x, t, Csat, KI):
    return KI * mytan((x * np.sqrt(t+1))/((Csat * (t+1))))

def quantile_integrator_log(
    scores,
    alpha,
    lr,
    Csat,
    KI,
    ahead,
    T_burnin,
    proportional_lr=True,
    *args,
    **kwargs
):
    data = kwargs['data'] if 'data' in kwargs.keys() else None
    results = quantile_integrator_log_scorecaster(scores, alpha, lr, data, T_burnin, Csat, KI, True, ahead, proportional_lr=proportional_lr, scorecast=False)
    results['method'] = "Quantile+Integrator (log)"
    return results


"""
    This is the master method for the quantile, integrator, and scorecaster methods.
"""
def quantile_integrator_log_scorecaster(
    scores,
    alpha,
    lr,
    data,
    T_burnin,
    Csat,
    KI,
    upper,
    ahead,
    integrate=True,
    proportional_lr=True,
    scorecast=True,
#    onesided_integrator=False,
    *args,
    **kwargs
):
    # Initialization
    T_test = scores.shape[0]
    qs = np.zeros((T_test,))
    qts = np.zeros((T_test,))
    integrators = np.zeros((T_test,))
    scorecasts = np.zeros((T_test,))
    covereds = np.zeros((T_test,))
    seasonal_period = kwargs.get('seasonal_period')
    if seasonal_period is None:
        seasonal_period = 1
    # Load the scorecaster
    try:
        # If the data contains a scorecasts column, then use it!
        if 'scorecasts' in data.columns:
            scorecasts = np.array([s[int(upper)] for s in data['scorecasts'] ])
            train_model = False
        else:
            scorecasts = np.load('./.cache/scorecaster/' + kwargs.get('config_name') + '_' + str(upper) + '.npy')
            train_model = False
    except:
        train_model = True
    # Run the main loop
    # At time t, we observe y_t and make a prediction for y_{t+ahead}
    # We also update the quantile at the next time-step, q[t+1], based on information up to and including t_pred = t - ahead + 1.
    #lr_t = lr * (scores[:T_burnin].max() - scores[:T_burnin].min()) if proportional_lr and T_burnin > 0 else lr
    for t in tqdm(range(T_test)):
        t_lr = t
        t_lr_min = max(t_lr - T_burnin, 0)
        lr_t = lr * (scores[t_lr_min:t_lr].max() - scores[t_lr_min:t_lr].min()) if proportional_lr and t_lr > 0 else lr
        t_pred = t - ahead + 1
        if t_pred < 0:
            continue # We can't make any predictions yet if our prediction time has not yet arrived
        # First, observe y_t and calculate coverage
        covereds[t] = qs[t] >= scores[t]
        # Next, calculate the quantile update and saturation function
        grad = alpha if covereds[t_pred] else -(1-alpha)
        #integrator = saturation_fn_log((1-covereds)[T_burnin:t_pred].sum() - (t_pred-T_burnin)*alpha, (t_pred-T_burnin), Csat, KI) if t_pred > T_burnin else 0
        integrator_arg = (1-covereds)[:t_pred].sum() - (t_pred)*alpha
        #if onesided_integrator:
        #    integrator_arg = np.clip(integrator_arg, 0, np.infty)
        integrator = saturation_fn_log(integrator_arg, t_pred, Csat, KI)
        # Train and scorecast if necessary
        if scorecast and train_model and t_pred > T_burnin and t+ahead < T_test:
            curr_scores = np.nan_to_num(scores[:t_pred])
            model = ThetaModel(
                    curr_scores.astype(float),
                    period=seasonal_period,
                    ).fit()
            scorecasts[t+ahead] = model.forecast(ahead)
        # Update the next quantile
        if t < T_test - 1:
            qts[t+1] = qts[t] - lr_t * grad
            integrators[t+1] = integrator if integrate else 0
            qs[t+1] = qts[t+1] + integrators[t+1]
            if scorecast:
                qs[t+1] += scorecasts[t+1]
    results = {"method": "Quantile+Integrator (log)+Scorecaster", "q" : qs}
    if train_model and scorecast:
        os.makedirs('./.cache/', exist_ok=True)
        os.makedirs('./.cache/scorecaster/', exist_ok=True)
        np.save('./.cache/scorecaster/' + kwargs.get('config_name') + '_' + str(upper) + '.npy', scorecasts)
    return results
