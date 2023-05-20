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
    asymmetric,
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
    order = kwargs["order"] if "order" in kwargs else None
    if log:
        data = np.log(data)
    print("Generating scores and forecasts...")
    for t in tqdm(range(T-1)):
        if log:
            if not asymmetric:
                scores[t+1] = np.abs(np.exp(forecasts[t]) - np.exp(data[t]))
            else:
                scores[t+1] = np.exp(forecasts[t]) - np.exp(data[t])
        else:
            if not asymmetric:
                scores[t+1] = np.abs(forecasts[t] - data[t])
            else:
                scores[t+1] = forecasts[t] - data[t]
        # Fit and predict with ARIMA
        if order is not None:
            if t > kwargs["T_burnin"]:
                if (arima_fit is None) or (t % kwargs["fit_every"] == 0):
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        arima_fit = ARIMA(data[max(t-kwargs['window_length'],0):t], order=order, enforce_stationarity=False, enforce_invertibility=False).fit(start_params=arima_fit.params if arima_fit is not None else None)
                    forecast_distance = min(kwargs["fit_every"], T - t - 1)
                    forecasts[t+1:t+forecast_distance+1] = arima_fit.forecast(forecast_distance)
        # Predict as piecewise constant
        else:
            forecasts[t+1] = data[t - t % kwargs["fit_every"]]

    np.savez(savename, scores=scores, forecasts=forecasts) if not log else np.savez(savename, scores=scores, forecasts=np.exp(forecasts))
    print("Finished generating scores and forecasts.")
    if log:
        return scores, np.exp(forecasts)
    else:
        return scores, forecasts
