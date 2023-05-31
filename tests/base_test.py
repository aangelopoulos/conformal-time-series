import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '../'))
import numpy as np
import pandas as pd
from core import standard_weighted_quantile, trailing_window, aci, quantile, quantile_integrator_log, quantile_integrator_log_scorecaster
from core.synthetic_scores import generate_scores
from core.model_scores import generate_forecasts
from datasets import load_dataset
import yaml
import pickle
import pdb

if __name__ == "__main__":
    json_name = sys.argv[1]
    if len(sys.argv) > 2:
        overwrite = sys.argv[2].split(",")
    else:
        overwrite = []
    args = yaml.safe_load(open(json_name))
    # Set up folder and filename
    foldername = './results/'
    config_name = json_name.split('.')[-2].split('/')[-1]
    filename = foldername + config_name + ".pkl"
    os.makedirs(foldername, exist_ok=True)
    real_data = args['real']
    quantiles_given = args['quantiles_given']
    multiple_series = args['multiple_series']
    score_function_name = args['score_function_name'] if real_data else "synthetic"
    asymmetric = False

    # Initialize the score function
    if real_data:
        score_function_name = args['score_function_name']
        if score_function_name == "absolute-residual":
            def score_function(y, forecast):
                return np.abs(y - forecast)
            def set_function(forecast, q):
                return np.array([forecast - q, forecast + q])
        elif score_function_name == "signed-residual":
            def score_function(y, forecast):
                return np.array([forecast - y, y - forecast])
            def set_function(forecast, q):
                return np.array([forecast - q[0], forecast + q[1]])
            asymmetric = True
        elif score_function_name == "cqr-symmetric":
            def score_function(y, forecasts):
                return np.maximum(forecasts[0] - y, y - forecasts[-1])
            def set_function(forecast, q):
                return np.array([forecast[0] - q, forecast[-1] + q])
        elif score_function_name == "cqr-asymmetric":
            def score_function(y, forecasts):
                return np.array([forecasts[0] - y, y - forecasts[-1]])
            def set_function(forecast, q):
                return np.array([forecast[0] - q[0], forecast[-1] + q[1]])
            asymmetric = True
        else:
            raise ValueError("Invalid score function name")

    # Get dataframe and add forecasts and scores to it
    if real_data:
        data = load_dataset(args['sequences'][0]['dataset'])
        # Get the forecasts
        if 'forecasts' not in data.columns:
            os.makedirs('./datasets/proc/', exist_ok=True)
            data_savename = './datasets/proc/' + config_name + '.npz'
            args['sequences'][0]['T_burnin'] = args['T_burnin']
            data['forecasts'] = generate_forecasts(data, data_savename, **args['sequences'][0])
        # Compute scores
        data['scores'] = [ score_function(y, forecast) for y, forecast in zip(data['y'], data['forecasts']) ]
    else:
        for key in args['sequences'].keys():
            scores_list += [generate_scores(**args['sequences'][key])]
        scores = np.concatenate(scores_list).astype(float)
        # Make a pandas dataframe with a datetime index and the scores in their own column called `scores'.
        data = pd.DataFrame({'scores': scores}, index=pd.date_range(start='1/1/2018', periods=len(scores), freq='D'))

    # Try reading in results
    try:
        with open(filename, 'rb') as handle:
            results = pickle.load(handle)
    except:
        results = {}

    # Loop through each method and learning rate, and compute the results
    for method in args['methods'].keys():
        if (method in results.keys()) and (method not in overwrite):
            continue
        fn = None
        if method == "Trail":
            fn = trailing_window
            args['methods'][method]['lrs'] = [None]
        elif method == "ACI":
            fn = aci
        elif method == "Quantile":
            fn = quantile
        elif method == "Quantile+Integrator (log)":
            fn = quantile_integrator_log
        elif method == "Quantile+Integrator (log)+Scorecaster":
            fn = quantile_integrator_log_scorecaster
        lrs = args['methods'][method]['lrs']
        kwargs = args['methods'][method]
        kwargs["T_burnin"] = args["T_burnin"]
        kwargs["data"] = data if real_data else None
        kwargs["seasonal_period"] = args["seasonal_period"] if "seasonal_period" in args.keys() else None
        kwargs["config_name"] = config_name
        # Compute the results
        results[method] = {}
        for lr in lrs:
            if asymmetric:
                stacked_scores = np.stack(data['scores'].to_list())
                q = [fn(stacked_scores[:,0], args['alpha']/2, lr, **kwargs)['q'], fn(stacked_scores[:,1], args['alpha']/2, lr, **kwargs)['q']]
                q = [ np.array([q[0][i], q[1][i]]) for i in range(len(q[0])) ]
            else:
                q = fn(data['scores'].to_numpy(), args['alpha'], lr, **kwargs)
            sets = [ set_function(data['forecasts'].to_numpy()[i], q[i]) for i in range(len(q)) ]
            results[method][lr] = { "q": q, "sets": sets }

    # Save some metadata
    results["scores"] = data['scores']
    results["alpha"] = args['alpha']
    results["T_burnin"] = args['T_burnin']
    results["quantiles_given"] = quantiles_given
    results["multiple_series"] = multiple_series
    results["real_data"] = real_data
    results["score_function_name"] = score_function_name
    results["asymmetric"] = asymmetric

    if real_data:
        results["forecasts"] = data['forecasts']
        results["data"] = data

    # Save results
    with open(filename, 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
