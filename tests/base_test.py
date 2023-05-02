import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '../'))
import numpy as np
import pandas as pd
from core import standard_weighted_quantile, trailing_window, aci, projected_gradient_descent, projected_gradient_descent_saturation, unconstrained, multiplicative_weights, mirror_descent, online_quantile, pi, arima_quantile, pid_gluon
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
    filename = foldername + json_name.split('.')[-2].split('/')[-1] + ".pkl"
    os.makedirs(foldername, exist_ok=True)
    real_data = args['real']
    quantiles_given = args['quantiles_given']
    multiple_series = args['multiple_series']

    # Compute scores
    scores_list = []
    forecasts_list = []
    data_list = []
    for key in args['sequences'].keys():
        if real_data: # Real data
            data = load_dataset(args['sequences'][key]['dataset'])
            if quantiles_given:
                scores, forecasts = np.maximum((data['lower'] - data['actual']).to_numpy(), (data['actual'] - data['upper']).to_numpy()), [data['lower'].to_numpy(), data['middle'].to_numpy(), data['upper'].to_numpy()]
                scores_list += [scores]
                forecasts_list += [forecasts]
                data_list += [data['actual'].to_numpy()]
            else:
                args['sequences'][key]['T_burnin'] = args['T_burnin']
                data_savename = './datasets/' + args['sequences'][key]['dataset'] + '.npz'
                scores, forecasts = generate_forecast_scores(data, data_savename, **args['sequences'][key])
                scores_list += [scores]
                forecasts_list += [forecasts]
                data_list += [data]
        else:
            scores_list += [generate_scores(**args['sequences'][key])]
    scores = np.concatenate(scores_list)

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
            results[method] = trailing_window(scores, args['alpha'], **(args['methods'][method]))
            continue
        elif method == "aci":
            fn = aci
        elif method == "pgd":
            fn = projected_gradient_descent
        elif method == "pgd+sat":
            fn = projected_gradient_descent_saturation
        elif method == "unconstrained":
            fn = unconstrained
        elif method == "quantile":
            fn = online_quantile
        elif method == "pi":
            fn = pi
        elif method == "pid+gluon":
            fn = pid_gluon
        elif method == "multiplicative":
            fn = multiplicative_weights
        elif method == "mirror":
            fn = mirror_descent
        elif method == "arima+quantile":
            fn = arima_quantile
        lrs = args['methods'][method]['lrs']
        kwargs = args['methods'][method]
        kwargs["T_burnin"] = args["T_burnin"]
        results[method] = { lr : fn(scores, args['alpha'], lr, **kwargs) for lr in lrs }
    results["scores"] = scores
    results["alpha"] = args['alpha']
    results["T_burnin"] = args['T_burnin']
    results["quantiles_given"] = quantiles_given
    results["multiple_series"] = multiple_series
    results["real_data"] = real_data

    if real_data:
        results["forecasts"] = np.concatenate(forecasts_list).T
        results["data"] = np.concatenate(data_list)

    # Save results
    with open(filename, 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
