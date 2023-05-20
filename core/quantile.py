import numpy as np
from scipy.optimize import brentq
import pdb

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
