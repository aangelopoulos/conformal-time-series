import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '../'))
import numpy as np
import pandas as pd
from core import standard_weighted_quantile, trailing_window, aci, online_quantile, pi, pid, pid_ets, pid_gluon
from core.synthetic_scores import generate_scores
from core.model_scores import generate_forecast_scores
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
    configname = json_name.split('.')[-2].split('/')[-1]
    filename = foldername + configname + ".pkl"
    os.makedirs(foldername, exist_ok=True)
    real_data = args['real']
    quantiles_given = args['quantiles_given']
    multiple_series = args['multiple_series']
    asymmetric = args['asymmetric'] if 'asymmetric' in args.keys() else False

    # Compute scores
    scores_list = []
    forecasts_list = []
    data_list = []
    for key in args['sequences'].keys():
        if real_data: # Real data
            data = load_dataset(args['sequences'][key]['dataset'])
            y = data[data['item_id'] == 'y']['target'].to_numpy()
            if quantiles_given:
                forecasts = data[data['item_id'] == 'forecast']['target']
                lower = np.array([forecast[0] for forecast in forecasts])
                middle = np.array([forecast[1] for forecast in forecasts])
                upper = np.array([forecast[-1] for forecast in forecasts])
                scores, forecasts = np.maximum(lower - y, y - upper), [lower, middle, upper]
                scores_list += [scores]
                forecasts_list += [forecasts]
                data_list += [data]
            else:
                args['sequences'][key]['T_burnin'] = args['T_burnin']
                data_savename = './datasets/' + configname + '.npz'
                scores, forecasts = generate_forecast_scores(y, asymmetric, data_savename, **args['sequences'][key])
                scores_list += [scores]
                forecasts_list += [forecasts]
                data_list += [data]
        else:
            scores_list += [generate_scores(**args['sequences'][key])]
    scores = np.concatenate(scores_list).astype(float)

    # Try reading in results
    try:
        with open(filename, 'rb') as handle:
            results = pickle.load(handle)
    except:
        results = {}

    # Compute results of methods
    for method in args['methods'].keys():
        if (method in results.keys()) and (method not in overwrite):
            continue
        fn = None
        if method == "trail":
            fn = trailing_window
            args['methods'][method]['lrs'] = [None]
        elif method == "aci":
            fn = aci
        elif method == "quantile":
            fn = online_quantile
        elif method == "pi":
            fn = pi
        elif method == "pid":
            fn = pid
        elif method == "pid+ets":
            fn = pid_ets
        elif method == "pid+gluon":
            fn = pid_gluon
        lrs = args['methods'][method]['lrs']
        kwargs = args['methods'][method]
        kwargs["T_burnin"] = args["T_burnin"]
        kwargs["data"] = data if real_data else None
        kwargs["seasonal_period"] = args["seasonal_period"] if "seasonal_period" in args.keys() else None
        kwargs["dataset_name"] = args['sequences'][key]['dataset']
        kwargs["config_name"] = configname
        if asymmetric:
            results[method] = { lr : [fn(-scores, args['alpha']/2, lr, **kwargs), fn(scores, args['alpha'], lr, **kwargs)] for lr in lrs }
        else:
            results[method] = { lr : fn(scores, args['alpha'], lr, **kwargs) for lr in lrs }
    results["scores"] = scores
    results["alpha"] = args['alpha']
    results["T_burnin"] = args['T_burnin']
    results["quantiles_given"] = quantiles_given
    results["multiple_series"] = multiple_series
    results["real_data"] = real_data
    results["asymmetric"] = asymmetric

    if real_data:
        results["forecasts"] = forecasts_list[0]
        results["data"] = data_list[0]

    # Save results
    with open(filename, 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
