import numpy as np

def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)