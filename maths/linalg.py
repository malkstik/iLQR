import numpy as np

def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)

if __name__ == "__main__":
    print(is_pos_def(-1*np.eye(3)))
