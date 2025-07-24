import pandas as pd
import numpy as np
import math

def log_likelihood(k,n):
    """Log-likelihood for a bucket."""
    if n == 0 or k == 0 or k == n:
        return 0
    p = k / n

    return k * math.log(p) + (n - k) * math.log(1 - p)

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
        df = data[['fico_score', 'default']].sort_values('fico_score').reset_index(drop=True)
        fico = df['fico_score'].values
        default = df['default'].values

        N = len(fico)

        #precompute prefixes sums
        prefix_defaults = np.cumsum(default)
        prefix_counts = np.arange(1, N + 1)

        #get stats between i and j
        def bucket_stats(i,j):
            k = prefix_defaults[j] - (prefix_defaults[i-1] if i > 0 else 0)
            n = j - i + 1
            return k, n

        dp = np.full((n_buckets+1, N), - np.inf)
        cuts = np.zeros((n_buckets+1, N), dtype=int)

        for j in range(N):
            k, n = bucket_stats(0, j)
            dp[1][j] = log_likelihood(k, n)

        for b in range(2, n_buckets+1):
            for j in range(b-1, N):
                best_val = -np.inf
                best_cut = -1
                for i in range(b-2, j):
                    k, n = bucket_stats(i+1, j)
                    val = dp[b-1][i] + log_likelihood(k, n)
                    if val > best_val:
                        best_val = val
                        best_cut = i
                
                dp[b][j] = best_val
                cuts[b][j] =  best_cut

        # Backtrack to find cut points
        boundaries = []
        end = N-1
        for b in range(n_buckets, 0, -1):
            start = cuts[b][end] + 1
            boundaries.append((fico[start], fico[end]))
            end = cuts[b][end]
        boundaries.reverse()
        return boundaries

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
    