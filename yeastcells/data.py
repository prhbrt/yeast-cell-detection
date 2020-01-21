import pickle
import numpy


def read_data(filename):
    with open(filename, 'rb') as f:
        X, y, z = pickle.load(f)
    X = X.reshape((-1,) + X.shape[2:])
    y = y.reshape((-1,) + y.shape[2:])
    z = z.reshape((-1,) + z.shape[2:])
        
    y = y > 0
    
    indices = y.mean(axis=1).mean(axis=1) < 0.1
    return X[indices], y[indices]
