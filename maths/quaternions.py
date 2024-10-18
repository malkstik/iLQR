import numpy as np
from scipy.spatial.transform import Rotation as R

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
    L_q[0, 1:] = -v
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

def QuaternionToParam(q: Quaternion) -> np.ndarray:
    '''
    Convert quaternion to 3 parameter representation
    :param q: quaternion of shape (4, num_quaternions)

    :return : 3 parameter quaternion representation of shape (3, num_quaternions)
    '''
    q /= np.linalg.norm(q) #ensure quaternions are of norm 1
    return q[1:]/q[0]

def QuaternionToRotation(q: Quaternion):
    '''
    Convert quaternion to rotation matrix
    '''
    L = GetLeftMatrix(q)
    return H.T @ T @ L @ T @ L @ H


def SampleQuaternion(near_identity = False):
    if near_identity:
        euler_angles = 0.5* np.random.randn(3)
        rotation = R.from_euler('xyz',
                                euler_angles, degrees= False)
    else:
        rotation = R.random()
    q = rotation.as_quat()
    return np.hstack((q[-1], q[1:])) 

if __name__ == "__main__":
    q = SampleQuaternion()  
    quat = np.hstack((q[-1], q[:-1]))

    rotation = R.from_quat(q)
    rotation_check = QuaternionToRotation(quat)

    print(rotation.as_matrix())
    print(rotation_check)

