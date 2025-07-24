import pandas as pd
import numpy as np

def generate_buckets(data, n_buckets=5, method='mse'):
    """
    Quantizes FICO scores into rating buckets
    data: DataFrame with columns ['fico_score', 'default']
    n_buckets: number of buckets
    method: 'mse' or 'loglikelihood'
    """

    scores = data['fico_score'].values
    defaults = data['default'].values

    if method == 'mse':
        percentiles = np.linspace(0, 100, n_buckets + 1)
        boundaries = np.percentile(scores, percentiles)
        boundaries[0] = 0
        boundaries[-1] = 850
        buckets = []

        for i in range(n_buckets):
            buckets.append((boundaries[i], boundaries[i + 1]))
        return buckets
    
    elif method == 'loglikelihood':
        raise NotImplementedError('Log-likelihood method not implemented yet')
    
    else:
        raise ValueError('Invalid method. Must be "mse" or "loglikelihood"')
    
def assign_rating(fico, buckets):
    """
    Assigns a rating (1 === best) absed on fico and buckets.
    """

    for idx, (low, high) in enumerate(buckets, start=1):
        if low <= fico <= high:
            return idx
        return len(buckets)
    