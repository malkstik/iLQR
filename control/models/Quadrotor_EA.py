
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
    def __init__(self, quadrotor: Diagram, multibody_plant: MultibodyPlant, 
                 Q: np.ndarray,
                 Qf: np.ndarray,
                 R: np.ndarray,
                 dt: int = 0.01,
                 N: int = 20,
                 **kwargs):
        self.nx = 12
        self.nu = 4
        
        self.quadrotor: Diagram = quadrotor
        self.plant: MultibodyPlant = multibody_plant
        self.mass = multibody_plant.CalcTotalMass(multibody_plant.GetMyContextFromRoot(quadrotor.CreateDefaultContext()))
        self.Q = Q
        self.Qf = Qf
        self.R = R
        self.dt = dt
        self.N = N

        self.xref = np.zeros(self.nx, dtype = np.float64)
        self.uref = np.zeros(self.nu, dtype = np.float64)

        self.CostDerivatives = autodiff.CostDerivatives(self._cost_stage,
                                                        self._cost_final,
                                                        self.xref,
                                                        self.uref,
                                                        self.nx,
                                                        self.nu)

    def discrete_dynamics(self, x, u):
        
        simulator = Simulator(self.quadrotor)
        simulator.Initialize()
        simulator_context = simulator.get_mutable_context()
        simulator_context.SetContinuousState(x)
        
        self.quadrotor.get_input_port().FixValue(simulator_context, u)
        sim_time = simulator_context.get_time()

        simulator.AdvanceTo(sim_time + self.dt)

        x_next = simulator_context.get_continuous_state_vector().CopyToVector()

        return x_next
    
    def set_references(self, xref, uref):
        self.xref = xref
        self.uref = uref
        self.CostDerivatives.set_references(xref, uref)

    def _linearize(self, ref_state: np.ndarray, ref_action: np.ndarray):
        '''
        Perform first order taylor approximation on system dynamics
        :param ref_state: array representing state to linearize about
        :param ref_action: array represneting action to linearize about
        :param ReduceState: bool representing whether to return A and B matrices that consider differential quaternions or not
        
        '''
        context = self.quadrotor.CreateDefaultContext()
        context.SetContinuousState(ref_state)
        
        self.quadrotor.get_input_port().FixValue(context, ref_action)
        sys =   FirstOrderTaylorApproximation(self.quadrotor, 
                                context,
                                self.quadrotor.get_input_port().get_index(),
                                self.quadrotor.get_output_port().get_index())
        

        return sys.A(), sys.B()

    def _batch_cost_stage(self, xtraj: np.ndarray, utraj: np.ndarray, xref: np.ndarray, uref: np.ndarray):
        '''
        Computes cost due to state and action trajectory

        :param xtraj: state trajectory, ndarray of shape (N-1, nx)
        :param utraj: action trajectory, ndarray of shape (N-1, nu)
        '''

        xerr =  xtraj - xref
        uerr = utraj - uref

        # Weighted errors
        weighted_state_errors = xerr @ self.Q  
        weighted_action = uerr@ self.R

        state_cost = np.sum(weighted_state_errors * xerr) 
        action_cost = np.sum(weighted_action * uerr)
        return 0.5*state_cost + 0.5* action_cost
    
    def _cost_stage(self, x: np.ndarray, u: np.ndarray, xref: np.ndarray, uref: np.ndarray):
        '''
        Computes cost due to a state action tuple

        :param x: state trajectory, ndarray of shape (N-1, nx)
        :param u: action trajectory, ndarray of shape (N-1, nu)
        '''        
        return 0.5*(x - xref).T @ self.Q @ (x - xref) + 0.5*(u - uref).T @ self.R @ (u - uref)

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

        J += self._batch_cost_stage(xtraj[:-1,:], utraj, self.xref, self.uref)
        J += self._cost_final(xf, self.xref)
        return J
    
    def stage(self, x, u):         
        l_x, l_u, l_xx, l_ux, l_uu = self.CostDerivatives.stage(x,u)
        # print("Autodiff")
        # print(f"l_x: {l_x}") 
        # print(f"l_u: {l_u}") 
        # print(f"l_xx: {l_xx}") 
        # print(f"l_uu: {l_uu}") 
        # print(f"l_ux: {l_ux}")   
        f_x, f_u = self._linearize(x, u)

        l_x = self.Q @ (x - self.xref)
        l_u = self.R @ u
        l_xx = self.Q
        l_uu = self.R
        l_ux = np.zeros((self.nu, self.nx))    

        # print("Handwritten")
        # print(f"l_x: {l_x}") 
        # print(f"l_u: {l_u}") 
        # print(f"l_xx: {l_xx}") 
        # print(f"l_uu: {l_uu}") 
        # print(f"l_ux: {l_ux}") 
        # print()

        return l_x, l_u, l_xx, l_ux, l_uu, f_x, f_u 
    
    def final(self, x):
        V_x, V_xx = self.CostDerivatives.final(x)
        return V_x, V_xx
    
class QuadrotorEAAnalyticModel(QuadrotorEAModel):
    def __init__(self, quadrotor: Diagram, multibody_plant: MultibodyPlant, 
                 Q: np.ndarray,
                 Qf: np.ndarray,
                 R: np.ndarray,
                 dt: int = 0.01,
                 N: int = 20,
                 **kwargs):
        super().__init__(quadrotor, multibody_plant, 
                       Q, Qf, R, dt, N, **kwargs)   
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


        # Compute Jacobians (A and B matrices)
        A = ca.jacobian(x_dot, x_sym)  # Jacobian w.r.t. state (x)
        B = ca.jacobian(x_dot, u_sym)  # Jacobian w.r.t. control input (u)


        self.xdot_fun = ca.Function('xdot', [x_sym, u_sym], [x_dot])
        # Now, you can create a CasADi function for A and B matrices
        self.A_fun = ca.Function('A', [x_sym, u_sym], [A])
        self.B_fun = ca.Function('B', [x_sym, u_sym], [B])

    def continuous_dynamics(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        '''
        :param x: state as [x, y, z, yaw, pitch, roll, x_dot, y_dot, z_dot, yaw_rate, pitch_rate, roll_rate]
        :param u: action as motor currents
        :return x_dot: state time derivative
        '''
        return np.array(self.xdot_fun(x, u).T.full())

    def discrete_dynamics(self, x0, u):
        # x_next = RK4(self.continuous_dynamics, x0, u, dt = self.dt)
        x_next = self.continuous_dynamics(x0, u)*self.dt + x0
        return x_next
    
    def _linearize(self, ref_state: np.ndarray, ref_action: np.ndarray):
        A = np.array(self.A_fun(ref_state, ref_action).full())
        B = np.array(self.B_fun(ref_state, ref_action).full())
        return A, B