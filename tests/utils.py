import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def moving_average(x, window=50):
    norm_factor = window / np.convolve( np.ones_like(x), np.ones(window), 'same' ) # Deal with edge effects
    return norm_factor * (np.convolve(x, np.ones(window), 'same') / window)
