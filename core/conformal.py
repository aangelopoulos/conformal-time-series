import numpy as np
from scipy.optimize import brentq
import pdb

def naive_conformal(args):
    Yhat_test, Y_test, alpha, T_burnin = args['Yhat_test'], args['Y_test'], args['alpha'], args['T_burnin']
    T_test = Yhat_test.shape[0]
    scores = np.abs(Yhat_test - Y_test)
    qhats = np.array( [np.quantile(scores[:t], np.clip(np.ceil((t+1)*(1-alpha))/t,0,1), method='higher') for t in range(T_burnin, T_test)] )
    prediction_sets = [Yhat_test[T_burnin:] - qhats, Yhat_test[T_burnin:] + qhats]
    covereds = ( Y_test[T_burnin:] >= prediction_sets[0] ) & ( Y_test[T_burnin:] <= prediction_sets[1] )
    return {'sets': prediction_sets, 'covereds': covereds, 'qhats': qhats}

def aci(args):
    Yhat_test, Y_test, alpha, lr_aci, T_burnin = args['Yhat_test'], args['Y_test'], args['alpha'], args['lr_aci'], args['T_burnin']
    scores = np.abs(Yhat_test - Y_test)
    T_test = scores.shape[0]

    # Get the ACI quantile at each time step
    alphas = np.ones((T_test,)) * alpha
    qhats = np.zeros((T_test,))
    covereds = np.zeros((T_test,)) > 0
    prediction_sets = [np.zeros((T_test,)), np.zeros((T_test,))]
    for t in range(T_burnin,T_test):
        alphas[t] = alphas[t-1] - lr_aci*((1-covereds[t-1].astype(float)) - alpha)
        conformal_level = np.clip(1-alphas[t],0,1)
        if alphas[t] < 1/(t+1):
            qhats[t] = np.inf
        else:
            qhats[t] = np.quantile(scores[:t], conformal_level, method='higher')
        prediction_sets[0][t] = Yhat_test[t] - qhats[t]
        prediction_sets[1][t] = Yhat_test[t] + qhats[t]
        covereds[t] = ( Y_test[t] >= prediction_sets[0][t] ) & ( Y_test[t] <= prediction_sets[1][t] )
    return {'sets': [prediction_sets[0][T_burnin:], prediction_sets[1][T_burnin:]], 'covereds': covereds[T_burnin:], 'alphas': alphas[T_burnin:], 'qhats': qhats[T_burnin:]}

# Quantile function
def standard_weighted_quantile(scores, wtildes, quantile, maxscore=np.infty, allow_illegal_weights=False):
    if not allow_illegal_weights:
        assert (np.array(wtildes).size == 0) or (wtildes.sum() <= 1.0 + 1e-4)
    if scores.shape[0] <= 5: # If not enough data, return infinite set
        return maxscore
    # For inversion
    idx_score_sort = np.argsort(scores)
    sorted_scores = scores[idx_score_sort]
    sorted_wtildes = wtildes[idx_score_sort]
    cumsum = np.cumsum(sorted_wtildes)
    if cumsum[-1] <= quantile: # If quantile asked for is too large, return infinite set
        return maxscore
    if cumsum[0] >= quantile:
        return sorted_scores[0] # If quantile asked for is too small, return smallest score
    def critical_point_quantile(q):
        if np.all(scores <= q):
            return 1.0 - quantile
        if np.all(scores >= q):
            return cumsum[0] - quantile
        return cumsum[np.maximum((sorted_scores <= q).sum()-1,0)] - quantile
    qhat = brentq(critical_point_quantile, scores.min()-10, scores.max()+10)
    rank = (sorted_scores <= qhat).sum()-1
    if cumsum[rank] < quantile:
        return sorted_scores[rank+1]
    else:
        return sorted_scores[rank]

def weighted_conformal(args):
    Yhat_test, Y_test, weights, alpha, T_burnin = args['Yhat_test'], args['Y_test'], args['fixed_weights'], args['alpha'], args['T_burnin']
    T_test = Yhat_test.shape[0]
    wtildes = weights/(weights.sum() + 1)
    scores = np.abs(Yhat_test - Y_test)
    qhats = np.array( [get_weighted_quantile(scores[t-weights.shape[0]:t], weights, 1-alpha) for t in range(T_burnin,T_test) ] )
    prediction_sets = [Yhat_test[T_burnin:] - qhats, Yhat_test[T_burnin:] + qhats]
    covereds = ( Y_test[T_burnin:] >= prediction_sets[0] ) & ( Y_test[T_burnin:] <= prediction_sets[1] )
    return {'sets': prediction_sets, 'covereds': covereds, 'qhats': qhats}
