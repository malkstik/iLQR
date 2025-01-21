import numpy as np

def is_pos_def(x: np.ndarray):
    lambdas = np.linalg.eigvals(x)
    # print(lambdas)
    return np.all(lambdas > 0)

def qr_inverse(A: np.ndarray) -> np.ndarray:
    '''
    Compute the inverse using QR decomposition

    :param A: some square matrix
    :returns : inverse of A
    '''
    
    Q, R = np.linalg.qr(A)
    return np.linalg.inv(R) @ Q.T


def is_singular(A) -> bool:
    '''
    Check for matrix singularity

    :param A: some square matrix

    '''
    rank = np.linalg.matrix_rank(A)
    return rank < A.shape[0]

def controllability_matrix(A, B):
    '''
    Computes the controllability matrix for a linear time-invariant system.

    :param A: The state matrix (nx x nx)
    :param B: The input matrix (nu x nu)

    Returns:
    The controllability matrix (nx x nx * nm)
    '''

    n = A.shape[0]
    C = B
    for i in range(1, n):
        C = np.hstack((C, np.linalg.matrix_power(A, i) @ B))
    return C

def is_controllable(A, B) -> bool:
    return not is_singular(controllability_matrix(A,B))


if __name__ == "__main__":
    print(is_pos_def(-1*np.eye(3)))
