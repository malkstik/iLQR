import numpy as np

def is_pos_def(x):
    lambdas = np.linalg.eigvals(x)
    # print(lambdas)
    return np.all(lambdas > 0)

if __name__ == "__main__":
    print(is_pos_def(-1*np.eye(3)))
