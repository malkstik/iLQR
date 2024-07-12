import numpy as np


Quaternion = np.ndarray
H = np.vstack((np.zeros((1,3)), np.eye(3)))
T = np.diag([1.0, -1.0, -1.0, -1.0])

def hat(q) -> np.ndarray:
    '''
    Applies hat operator to vector portion of quaternion
    :param q: quaternion
    :return qhat: quaternion vector portion promoted to skew symmetric matrix
    '''
    v = q[1:].squeeze().tolist()
    qhat = np.array([[ 0.0, -v[2],  v[1]],
                     [v[2],   0.0, -v[0]],
                     [-v[1],  v[0],  0.0]])
    return qhat

def GetLeftMatrix(q) -> np.ndarray:
    '''
    :param q: quaternion
    :return L_q: Left multiplication matrix corresponding to q of shape (4,4)
    '''
    w = q[0]
    v = q[1:].squeeze()

    L_q = np.zeros((4,4))
    L_q[0, 0] = w
    L_q[0, 1:] = v
    L_q[1:, 0] = v
    L_q[1:, 1:] = w* np.eye(3) + hat(q)

    return L_q

def GetAttititudeJacobian(q):
    '''
    Compute attitude jacobian
    :param q: quaternion
    :return G: attitude jacobian, shape (4,3)
    '''
    G = GetLeftMatrix(q) @ H
    return G

def ParamToQuaternion(phi):
    '''
    Convert 3 parameter representation to quaternion
    '''
    return 1/np.sqrt(1 + np.dot(phi, phi)) * np.hstack((np.ones(1), phi.reshape((3,1))))

def QuaternionToParam(q: Quaternion):
    '''
    Convert quaternion to 3 parameter representation
    '''
    return q[1:]/q[0]

def QuaternionToRotation(q: Quaternion):
    '''
    Convert quaternion to rotation matrix
    '''
    L = GetLeftMatrix(q)
    return H.T @ T @ L @ T @ L @ H