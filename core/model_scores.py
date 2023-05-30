import numpy as np
import warnings
from .ar import generate_process
from statsmodels.tsa.arima.model import ARIMA
import pdb
from tqdm import tqdm
"""
    Generates forecasts from an ARIMA model
"""
def generate_forecasts(
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
            forecasts = saved["forecasts"]
            return forecasts
        except:
            pass
    T = data.shape[0]
    forecasts = np.zeros((T,))
    arima_fit = None
    order = kwargs["order"]
    y = data['y'] if not log else np.log(data['y'])

    # Generate the forecasts
    print("Generating forecasts...")
    for t in tqdm(range(T-1)):
        # Fit and predict with ARIMA
        if t > kwargs["T_burnin"]:
            if (arima_fit is None) or (t % kwargs["fit_every"] == 0):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    arima_fit = ARIMA(y[max(t-kwargs['window_length'],0):t], order=order, enforce_stationarity=False, enforce_invertibility=False).fit(start_params=arima_fit.params if arima_fit is not None else None)
                forecast_distance = min(kwargs["fit_every"], T - t - 1)
                forecasts[t+1:t+forecast_distance+1] = arima_fit.forecast(forecast_distance)
    if log:
        forecasts = np.exp(forecasts)
    print("Finished generating forecasts.")

    # Save and return
    np.savez(savename, forecasts=forecasts)
    return forecasts
