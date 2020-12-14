import skfuzzy as fuzz
import numpy as np
import pandas as pd
import sys


np.set_printoptions(threshold=sys.maxsize)


'''
All funstion return a list of value and a list of tresholds
example:

'''


def linear(df, n=5):
    values = [min(df)]
    for i in range(1, n):
        values.append( (max(df)-min(df))/n*i)
    values.append(max(df))
    return values

def uniform(df, n=5):
    df = df.sort_values(ignore_index=True)
    values = [min(df)]
    for i in range(1,n):
        values.append(len(df.index)/n * i)
    values.append(max(df))
    return values

def cMeans(df, n=5):
    # data reshaping
    od = df.sort_values(ignore_index=True)
    data = np.vstack((od, od))
    # data classification
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(data, n, 2, error=0.005, maxiter=1000, init=None)
    # point class
    clusters = np.argmax(u, axis=0)
    # where class changes
    border_index = np.where(clusters[:-1] != clusters[1:])[0]
    # compute tresholds
    values = [min(df)]
    for b in border_index:
        values.append( (od[b]+od[b+1])/2)
    values.append(max(df))
    return values
















'''
source: https://pythonhosted.org/scikit-fuzzy/api/skfuzzy.cluster.html
fuzz.cluster.means() returns:

    cntr : 2d array, size (S, c)
    Cluster centers. Data for each center along each feature provided for every cluster (of the c requested clusters).

    u : 2d array, (S, N)
    Final fuzzy c-partitioned matrix.

    u0 : 2d array, (S, N)
    Initial guess at fuzzy c-partitioned matrix (either provided init or random guess used if init was not provided).

    d : 2d array, (S, N)
    Final Euclidian distance matrix.

    jm : 1d array, length P
    Objective function history.

    p : int
    Number of iterations run.

    fpc : float
    Final fuzzy partition coefficient.

    from : https://pythonhosted.org/scikit-fuzzy/api/skfuzzy.cluster.html
'''
