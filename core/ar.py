import numpy as np

def generate_process(T, beta, std, theta=None, X=None):
    root_mags = np.abs(np.roots([1,] + list(-beta)))
    assert np.all(root_mags <= 1.)
    if ((theta is None) or (X is None)):
        theta = np.zeros((1,))
        X = np.zeros((T,1))
    Y = np.zeros((T,))
    p = beta.shape[0]
    for t in range(T):
        for i in range(p):
            if t-(i+1) < 0:
                continue
            Y[t] += beta[i]*Y[t-(i+1)]
        Y[t] += np.random.normal(scale=std)
        Y[t] += theta.dot(X[t,:])
    return Y

def generate_process_heteroskedastic(T, beta, std, theta=None, nu=None, X=None):
    root_mags = np.abs(np.roots([1,] + list(-beta)))
    assert np.all(root_mags <= 1.)
    assert not (nu is None)
    if ((theta is None) or (X is None)):
        theta = np.zeros((1,))
        X = np.zeros((T,1))
    Y = np.zeros((T,))
    p = beta.shape[0]
    for t in range(T):
        for i in range(p):
            if t-(i+1) < 0:
                continue
            Y[t] += beta[i]*Y[t-(i+1)]
        Y[t] += nu.dot(X[t,:]) * np.random.normal(scale=std)
        Y[t] += theta.dot(X[t,:])
    return Y

def fit_ar_model(Y,p,X=None):
    T = Y.shape[0]
    M = np.zeros((T-p,p))
    Yflip = np.flip(Y)
    for i in range(0,T-p):
        M[i,:] = Yflip[i+1:i+1+p]
    M = np.flip(M, axis=0)
    M = np.flip(M, axis=1)
    if X is None:
        betahat = np.flip(np.linalg.pinv(M)@Y[p:])
        return betahat
    else:
        d = X.shape[1]
        M = np.concatenate([M,X[p:]],axis=1)
        out = np.flip(np.linalg.pinv(M)@Y[p:])
        betahat = out[d:]
        thetahat = out[:d]
        return betahat, thetahat
    
def predict(Y_test, betahat, thetahat=None, X_test=None):
    T_test = Y_test.shape[0]
    p = betahat.shape[0]
    if (thetahat is None) or (X_test is None):
        thetahat = np.zeros((1,))
        X_test = np.zeros((T_test,1))
    Yhat = np.zeros((T_test,))
    for i in range(p,T_test):
        Yhat[i] = betahat.dot(np.flip(Y_test[i-p:i])) + thetahat.dot(X_test[i])
    return Yhat