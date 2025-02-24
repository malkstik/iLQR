import numpy as np
import casadi as ca

# Quaternion functions using CasADi
H = np.vstack((np.zeros((1,3)), np.eye(3)))
T = np.diag([1.0, -1.0, -1.0, -1.0])
quat = ca.MX.sym('quat', 4)
qw, qx, qy, qz = quat/ca.norm_2(quat[:4])
params = ca.MX.sym('phi', 3) # Three parameter representation

# Define rotation matrix R_NB from body to world frame
LeftMatrix = ca.vertcat(
    ca.horzcat( qw,  qx,  qy,  qz),
    ca.horzcat(-qx,  qw, -qz,  qy),
    ca.horzcat(-qy,  qz,  qw, -qx),
    ca.horzcat(-qz, -qy,  qx,  qw)
)
AttitudeJacobian = LeftMatrix @ H
Conjugate = ca.horzcat(qw, -qx, -qy, -qz)
ParamConversion = 1/ca.sqrt(1 + params.T @ params) * ca.horzcat(1, params)
QuatConversion = ca.horzcat(qx, qy, qz)/qw
Quat2Rot = H.T @ T @ LeftMatrix @ T @ LeftMatrix @ H

GetLeftMatrix = ca.Function("GetLeftMatrix", [quat], [LeftMatrix])
GetAttitudeJacobian = ca.Function("GetAttitudeJacobian", [quat], [AttitudeJacobian])
GetConjugate= ca.Function("GeConjugate", [quat], [Conjugate])

ParamToQuaternion = ca.Function("ParamToQuaternion", [params], [ParamConversion])
QuaternionToParam = ca.Function("QuaternionToParam", [quat], [QuatConversion])
QuaternionToRotation = ca.Function("QuaternionToRotation", [quat], [Quat2Rot])
