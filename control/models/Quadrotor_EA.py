
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
from math import sin, cos
import casadi as ca

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))
from maths import autodiff
from maths.forward_integration import RK4
from time import time
    
class QuadrotorEAModel(Model):
    def __init__(self,
                 Q: np.ndarray,
                 Qf: np.ndarray,
                 R: np.ndarray,
                 dt: int = 0.01,
                 N: int = 20,
                 integration: str = "RK4",
                 **kwargs):

        self.nx = 12
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
        self.L = 0.15  # Length of the arms (m).
        self.kF = 1.0  # Force input constant.
        self.kM = 0.0245  # Moment input constant.
        self.m = 0.775
        self.MoI = np.array([[0.0015, 0, 0,],
                             [0, 0.0025, 0],  
                             [0, 0, 0.0035]])
        self.inv_MoI = np.linalg.inv(self.MoI)

        # Define symbolic variables
        x_sym = ca.MX.sym('state', 12)  # State [x, y, z, yaw, pitch, roll, x_dot, y_dot, z_dot, yaw_rate, pitch_rate, roll_rate]
        u_sym = ca.MX.sym('action', 4)   # Control inputs (motor currents)

        # Extract parts of the state vector
        euler_angles = x_sym[3:6]
        linear_velocity = x_sym[6:9]
        angular_velocity = x_sym[9:]
        rpy_angular_velocity = ca.vertcat(angular_velocity[2], angular_velocity[1], angular_velocity[0])

        # Define rotation matrix R_NB from body to world frame
        y, p, r = euler_angles[0], euler_angles[1], euler_angles[2]
        cr, cp, cy = ca.cos(r), ca.cos(p), ca.cos(y)
        sr, sp, sy = ca.sin(r), ca.sin(p), ca.sin(y)

        R_NB = ca.vertcat(
            ca.horzcat(cp*cy, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr),
            ca.horzcat(sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr),
            ca.horzcat(  -sp,       cp*sr,             cr*cp)
        )
        # Compute thrust due to rotors
        uF_Bz = self.kF * u_sym
        # Faero_B = ca.MX([0, 0, uF_Bz[0] + uF_Bz[1] + uF_Bz[2]])
        Faero_B = ca.MX.zeros(1, 3)
        Faero_B[0,-1] = uF_Bz[0] + uF_Bz[1] + uF_Bz[2] + uF_Bz[3]

        # Compute gravity and world frame
        Fgrav_N = ca.MX([0, 0, -self.m * 9.81])

        # Total forces
        Ftot_N = Fgrav_N + R_NB @ Faero_B.T
        linear_accel = Ftot_N / self.m

        # Compute moments
        Mx_B = self.L * (uF_Bz[1] - uF_Bz[3])
        My_B = self.L * (uF_Bz[2] - uF_Bz[0])
        uTau_Bz = self.kM * u_sym
        Mz_B = uTau_Bz[0] - uTau_Bz[1] + uTau_Bz[2] - uTau_Bz[3]
        Tau_B = ca.horzcat(Mx_B, My_B, Mz_B)

        # Base angular velocity in world
        M = ca.vertcat(
            ca.horzcat(1., 0., -sp),
            ca.horzcat(0., cr, sr*cp),
            ca.horzcat(0., -sr, cr*cp)
        )
        w_BN_B = M @ rpy_angular_velocity

        # Angular acceleration in body frame
        alpha_NB_N = R_NB @ self.inv_MoI @ (Tau_B - ca.cross(w_BN_B, self.MoI @ w_BN_B).T).T

        # Compute angular acceleration in world coordinates
        Minv = ca.vertcat(
            ca.horzcat(cy/cp, sy/cp, 0.0),
            ca.horzcat(-sy, cy, 0.0),
            ca.horzcat(cy/cp*sp, sy/cp*sp, 1.0)
        )

        angular_accel = Minv @ alpha_NB_N

        yaw_rate, pitch_rate = angular_velocity[0], angular_velocity[1]

        Mdt = ca.vertcat(
            ca.horzcat(-cy*sp*pitch_rate - sy*cp*yaw_rate,   -cy*yaw_rate,    0.0),
            ca.horzcat(-sy*sp*pitch_rate + cy*cp*yaw_rate,   -sy*yaw_rate,    0.0),
            ca.horzcat( -cp*pitch_rate,                 0.0,        0.0)
        )

        rpy_angular_accel = Minv @ (alpha_NB_N - Mdt @ rpy_angular_velocity)
        angular_accel = ca.vertcat(rpy_angular_accel[2], rpy_angular_accel[1], rpy_angular_accel[0])

        # Define the system dynamics
        x_dot = ca.vertcat(linear_velocity, angular_velocity, linear_accel, angular_accel)
        xdot_fun = ca.Function('xdot', [x_sym, u_sym], [x_dot])

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

    def _linearize_discrete(self, ref_state, ref_action):
        discreteA, discreteB = self.A_discrete_fun(ref_state, ref_action), self.B_discrete_fun(ref_state, ref_action) 
        return discreteA, discreteB

    def _linearize_continuous(self,ref_state, ref_action):
        contA, contB = self.A_continuous_fun(ref_state, ref_action), self.B_continuous_fun(ref_state, ref_action) 
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