

from control.models.model import Model


from pydrake.all import (Diagram,
                         MultibodyPlant,
                         FirstOrderTaylorApproximation,
                         Simulator,
                         AutoDiffXd)
import pydrake.symbolic as sym
import numpy as np 
import os
import sys
import casadi as ca
from maths.casadi_quaternions import (GetLeftMatrix, 
                                      GetAttitudeJacobian, 
                                      GetConjugate, 
                                      ParamToQuaternion, 
                                      QuaternionToParam, 
                                      QuaternionToRotation
)

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))
from maths import autodiff
from maths.forward_integration import RK4
from time import time
    
class QuadrotorQuatModel(Model):
    '''
    Quadrotor model class defining defining dynamics of quadrotor with states 
    [qw, qx, qy, qz, x, y, z, wx, wy, wz, vx, vy, vz]^T

    and stage cost
    (x-xref).T @ G(q).T @ Q @ (x-xref)@G(q) + u.T@R@u
    '''
    def __init__(self,
                 Q: np.ndarray,
                 Qf: np.ndarray,
                 R: np.ndarray,
                 dt: int = 0.01,
                 N: int = 20,
                 integration: str = "RK4",
                 **kwargs):

        self.nx = 13
        self.nu = 4
        
        self.Q = Q
        self.Qf = Qf
        self.R = R
        self.dt = dt
        self.N = N

        self.xref = np.zeros(self.nx, dtype = np.float64)
        self.uref = np.zeros(self.nu, dtype = np.float64)

        # Quadrotor params
        # Kinda lazy to hardcode but what can you do ¯\_(ツ)_/¯
        L = 0.15  # Length of the arms (m).
        kF = 1.0  # Force input constant.
        kM = 0.0245  # Moment input constant.
        m = 0.775
        MoI = np.array([[0.0015, 0, 0,],
                            [0, 0.0025, 0],  
                            [0, 0, 0.0035]])
        inv_MoI = np.linalg.inv(MoI)

        # Define symbolic variables
        # State [qw, qx, qy, qz, x, y, z, wx, wy, wz, vx, vy, vz]^T
        x_sym = ca.MX.sym('state', 13)  
        u_sym = ca.MX.sym('action', 4)   # Control inputs (motor currents)

        # Extract parts of the state vector
        quat = x_sym[:4] #Describes rotation from world to quadrotor base frame
        position = x_sym[4:7] #Describes position of quadrotor in world frame
        angular_velocity = x_sym[7:10] #Describes rotational rates of quadrotor in world frame
        linear_velocity = x_sym[10:] #Describes linear velocities of quadrotor in world frame

        R_NB = QuaternionToRotation(quat)
        w_BN_B = R_NB.T @ angular_velocity

        quat_dot = 0.5 * GetAttitudeJacobian(quat) @ w_BN_B

        # Compute thrust due to rotors
        uF_Bz = kF * u_sym
        # Faero_B = ca.MX([0, 0, uF_Bz[0] + uF_Bz[1] + uF_Bz[2]])
        Faero_B = ca.MX.zeros(1, 3)
        Faero_B[0,-1] = uF_Bz[0] + uF_Bz[1] + uF_Bz[2] + uF_Bz[3]

        # Compute gravity and world frame
        Fgrav_N = ca.MX([0, 0, -m * 9.81])

        # Total forces
        Ftot_N = Fgrav_N + R_NB @ Faero_B.T
        linear_accel = Ftot_N / m

        # Compute moments
        Mx_B = L * (uF_Bz[1] - uF_Bz[3])
        My_B = L * (uF_Bz[2] - uF_Bz[0])
        uTau_Bz = kM * u_sym
        Mz_B = uTau_Bz[0] - uTau_Bz[1] + uTau_Bz[2] - uTau_Bz[3]
        Tau_B = ca.horzcat(Mx_B, My_B, Mz_B)

        # Angular acceleration in body frame
        angular_accel = R_NB @ inv_MoI @ (Tau_B - ca.cross(w_BN_B, MoI @ w_BN_B).T).T

        x_dot = ca.vertcat(quat_dot, linear_velocity, angular_accel, linear_accel)

        xdot_fun = ca.Function("xdot", [x_sym, u_sym], [x_dot])

        # RK4 integration
        if integration == "RK4":
            print("Integrating dynamics with RK4")
            f1 = xdot_fun(x_sym.T, u_sym).T
            f2 = xdot_fun(x_sym.T + 0.5*f1*dt, u_sym).T
            f3 = xdot_fun(x_sym.T + 0.5*f2*dt, u_sym).T
            f4 = xdot_fun(x_sym.T + 1.0*f3*dt, u_sym).T
            x_next = x_sym.T + (f1 + 2*f2 + 2*f3 + f4)*dt/6.0
        # Euler integration
        else:
            print("Integrating dynamics with Euler")
            x_next = x_sym.T + xdot_fun(x_sym.T, u_sym).T * dt

        self.x_next = ca.Function('xnext', [x_sym, u_sym], [x_next])
        # Compute Jacobians (A and B matrices)
        A_continuous = ca.jacobian(x_dot, x_sym)
        B_continuous = ca.jacobian(x_dot, u_sym)
        A_discrete = ca.jacobian(x_next, x_sym)  
        B_discrete = ca.jacobian(x_next, u_sym)  

        # CasADi function for A and B matrices
        self.A_discrete_fun = ca.Function('A_discrete', [x_sym, u_sym], [A_discrete])
        self.B_discrete_fun = ca.Function('B_discrete', [x_sym, u_sym], [B_discrete])      
        self.A_continuous_fun = ca.Function('A_continuous', [x_sym, u_sym], [A_continuous])
        self.B_continuous_fun = ca.Function('B_continuous', [x_sym, u_sym], [B_continuous])             

    def set_references(self, xref, uref):
        self.xref = xref
        self.uref = uref

    def discrete_dynamics(self, x, u):   
        x_next = self.x_next(x, u)
        return x_next

    def _cost_stage(self, x: np.ndarray, u: np.ndarray, xref: np.ndarray, uref: np.ndarray):
        '''
        Computes cost due to a state action tuple

        :param x: state trajectory, ndarray of shape (N-1, nx)
        :param u: action trajectory, ndarray of shape (N-1, nu)
        '''        
        xerr = (x - xref).reshape((12,1))
        uerr = u - uref

        state_cost = 0.5*(xerr).T @ self.Q @ (xerr) 
        action_cost = 0.5*(uerr).T @ self.R @ (uerr)
        return state_cost + action_cost

    def _cost_final(self, xf: np.ndarray, xref: np.ndarray):
        '''
        Computes cost due to final state of state trajectory
        '''
        return 0.5*(xf - xref).T @ self.Qf @ (xf - xref)
    
    def cost_trj(self, xtraj: np.ndarray, utraj: np.ndarray):
        '''
        Computes total cost of trajectory
        :param xtraj: state trajectory, ndarray of shape (num_time_steps, nx)
        :param utraj: action trajectory, ndarray of shape (num_time_steps-1, nu)

        '''
        J = 0
        xf = xtraj[-1, :]


        for n in range(self.N-1):
            J+= self._cost_stage(xtraj[n,:], utraj[n,:], self.xref, self.uref)
        J += self._cost_final(xf, self.xref)

        return J
    
    def _GetStateDiffJacobian(self, q0):
        StateDiffJacobian = np.zeros((13, 12), dtype = q0.dtype)
        StateDiffJacobian[:4, :3] = GetAttitudeJacobian(q0)
        StateDiffJacobian[4:, 3:] = np.eye(9)

        return StateDiffJacobian

    def CalcDifferentialState(self, state: np.ndarray, ref_state: np.ndarray):
        '''
        Computes differential state accounting for quaternion kinematics

        :param state: ndarray of shape (nx)
        :param ref_state: ndarray of shape (nx,)

        :return: ndarray of shape (nx-1)
        '''

        q_ref = ref_state[:4]
        q = state[:4]

        quaternion_error = GetLeftMatrix(q_ref).T @ q.T

        # if(np.any(quaternion_error[0]) <= 0.001):
        #     print(f"q_ref: {q_ref}\n q: {q}")

        differential_quaternion = np.array(QuaternionToParam(quaternion_error)).squeeze()

        return np.hstack((differential_quaternion, state[4:] - ref_state[4:]))


    def _linearize_discrete(self, ref_state, ref_action):
        A, B = self.A_discrete_fun(ref_state, ref_action), self.B_discrete_fun(ref_state, ref_action) 
        q0 = ref_state[:4]
        StateDiffJacobian = self._GetStateDiffJacobian(q0)

        discreteA = StateDiffJacobian.T @ A @ StateDiffJacobian
        discreteB = StateDiffJacobian.T @ B

        return discreteA, discreteB

    def _linearize_continuous(self,ref_state, ref_action):
        A, B = self.A_continuous_fun(ref_state, ref_action), self.B_continuous_fun(ref_state, ref_action) 
        q0 = ref_state[:4]
        StateDiffJacobian = self._GetStateDiffJacobian(q0)

        contA = StateDiffJacobian.T @ A @ StateDiffJacobian
        contB = StateDiffJacobian.T @ B        
        return contA, contB
    
    def stage(self, x, u):         
        f_x, f_u = self._linearize_discrete(x, u)

        l_x = self.Q @ (x - self.xref)
        l_u = self.R @ (u - self.uref)
        l_xx = self.Q
        l_uu = self.R
        l_ux = np.zeros((self.nu, self.nx))    

        return l_x, l_u, l_xx, l_ux, l_uu, f_x, f_u 
    
    def final(self, x):
        V_xx = self.Qf
        V_x = self.Qf @ (x - self.xref)
        return V_x, V_xx