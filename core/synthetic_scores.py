import numpy as np
from .ar import generate_process
import pdb
"""
    Generates scores from a pre-defined set of categories.
"""
def generate_scores(
    category,
    *args,
    **kwargs
    ):
    if category == "linear":
        return linear_scores(**kwargs)
    elif category == "sinusoid":
        return sinusoidal_scores(**kwargs)
    elif category == "autoregressive":
        return generate_process(kwargs['length'], np.array(kwargs['beta']), kwargs['sigma']) + kwargs['start_point']
    else:
        raise Exception("Score category not implemented!")

def linear_scores(
    start_point,
    end_point,
    length,
    sigma,
    *args,
    **kwargs
    ):
    return np.linspace(start_point, end_point, length) + np.random.normal(loc=0.0, scale=sigma, size=(length,))

def sinusoidal_scores(
    period,
    minimum,
    maximum,
    sigma,
    length,
    *args,
    **kwargs
    ):
    time = np.arange(length)
    c = 2*np.pi/period
    return (np.sin(c*time) + 0.5)*maximum + minimum + np.random.normal(loc=0.0, scale=sigma, size=(length,))
