# Code for statistics to evaluate our algorithms

import numpy as np

def percent_error(actual, predicted):
    actual = np.array(actual)
    predicted = np.array(predicted)
    return abs(100*(actual - predicted) / actual)

def mean_absolute_error(actual, predicted):
    actual = np.array(actual)
    predicted = np.array(predicted)
    return np.mean(np.abs(actual - predicted))
