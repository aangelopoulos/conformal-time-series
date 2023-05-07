import numpy as np
import warnings
from .ar import generate_process
from statsmodels.tsa.arima.model import ARIMA
import pdb
from tqdm import tqdm
"""
    Generates scores from a pre-defined set of categories.
"""
def generate_forecast_scores(
    data,
    savename,
    overwrite,
    log,
    *args,
    **kwargs
    ):
    if not overwrite:
        try:
            saved = np.load(savename)
            scores = saved["scores"]
            forecasts = saved["forecasts"]
            return scores, forecasts
        except:
            pass
    T = data.shape[0]
    scores = np.zeros((T,))
    forecasts = np.zeros((T,))
    arima_fit = None
    if log:
        data = np.log(data)
    print("Generating scores and forecasts...")
    for t in tqdm(range(T-1)):
        if log:
            scores[t+1] = np.abs(np.exp(forecasts[t]) - np.exp(data[t]))
        else:
            scores[t+1] = np.abs(forecasts[t] - data[t])
        if (t > kwargs["T_burnin"]):
            #if (arima_fit is None) or (t % kwargs["fit_every"] == 0):
            #    with warnings.catch_warnings():
            #        warnings.simplefilter("ignore")
            #        arima_fit = ARIMA(data[max(t-kwargs['window_length'],0):t], order=kwargs['order'], enforce_stationarity=False, enforce_invertibility=False).fit(start_params=arima_fit.params if arima_fit is not None else None)
            #forecasts[t+1] = arima_fit.forecast()[0]
            if (t % kwargs["fit_every"] == 0):
                forecasts[t+1] = data[t]
            else:
                forecasts[t+1] = forecasts[t]
        else:
            forecasts[t+1] = data[t]
    np.savez(savename, scores=scores, forecasts=forecasts)
    print("Finished generating scores and forecasts.")
    if log:
        return scores, np.exp(forecasts)
    else:
        return scores, forecasts
