import numpy as np
import random

class SynteticData:

    def generate_data(self, clusters=8, num_points=400, mean=None, covariance=None, interval=None):
        assert num_points%clusters == 0
        X = np.zeros((num_points,2))  # Only 2-D data is generated

        if mean is None:
            mean = np.random.random((clusters, 2)) * clusters * 1.5

        chunk = int(num_points/clusters)

        for c in range(clusters):
            m = mean[c]
            std = (np.random.rand(2)+ 0.1) #np.random.normal(0,2,2)
            X[c*chunk:(c+1)*chunk, 0] = np.random.normal(m[0], std[0], chunk)
            X[c*chunk:(c+1)*chunk, 1] = np.random.normal(m[1], std[1], chunk)
        return X