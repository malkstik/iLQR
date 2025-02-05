
from control.models.model import Model


from pydrake.all import (Diagram,
                         MultibodyPlant,
                         FirstOrderTaylorApproximation,
                         Simulator)
import pydrake.symbolic as sym
import numpy as np 
import os
import sys
from math import sin, cos

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))
from maths import autodiff
from maths.forward_integration import RK4

class QuadrotorEAModel(Model):
    def __init__(self, quadrotor: Diagram, multibody_plant: MultibodyPlant, 
                 Q: np.ndarray,
                 Qf: np.ndarray,
                 R: np.ndarray,
                 dt: int = 0.01,
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

    def _cost_stage(self, x: np.ndarray, u: np.ndarray, xref: np.ndarray, uref: np.ndarray):
        '''
        Computes cost due to state and action trajectory

        :param xtraj: state trajectory, ndarray of shape (N, nx)
        :param utraj: action trajectory, ndarray of shape (N-1, nu)
        '''

        xerr =  x - xref
        uerr = u - uref


        # Weighted errors
        weighted_state_errors = xerr @ self.Q  
        weighted_action = uerr@ self.R

        state_cost = np.sum(weighted_state_errors * xerr) 
        action_cost = np.sum(weighted_action * uerr)

        return 0.5*state_cost + 0.5* action_cost
    
    def _cost_final(self, xf: np.ndarray, xref: np.ndarray):
        '''
        Computes cost due to final state of state trajectory
        '''
        xerr = (xf - xref).T    
        weighted_state_error = xerr @ self.Qf

        return 0.5 * np.sum(weighted_state_error* xerr)
    
    def cost_trj(self, xtraj: np.ndarray, utraj: np.ndarray):
        '''
        Computes total cost of trajectory
        :param xtraj: state trajectory, ndarray of shape (num_time_steps, nx)
        :param utraj: action trajectory, ndarray of shape (num_time_steps-1, nu)

        '''
        J = 0
        xf = xtraj[-1, :]

        J += self._cost_stage(xtraj[:-1,:], utraj[:-1,:], self.xref, self.uref)
        J += self._cost_final(xf, self.xref)
        return J
    
    def stage(self, x, u):
        l_x, l_u, l_xx, l_ux, l_uu = self.CostDerivatives.stage(x,u)
        f_x, f_u = self._linearize(x, u)
        return l_x, l_u, l_xx, l_ux, l_uu, f_x, f_u 
    
    def final(self, x):
        V_x, V_xx = self.CostDerivatives.final(x)
        return V_x, V_xx
    
class QuadrotorEAAnalyticModel(Model):
    def __init__(self, quadrotor: Diagram, multibody_plant: MultibodyPlant, 
                 Q: np.ndarray,
                 Qf: np.ndarray,
                 R: np.ndarray,
                 dt: int = 0.01,
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

        self.xref = np.zeros(self.nx, dtype = np.float64)
        self.uref = np.zeros(self.nu, dtype = np.float64)
        self.ModelAutoDiff = autodiff.ModelAutoDiff(self.discrete_dynamics,
                                                        self._cost_stage,
                                                        self._cost_final,
                                                        self.xref,
                                                        self.uref,
                                                        self.nx,
                                                        self.nu)

    def continuous_dynamics(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        '''
        :param x: state as [x, y, z, yaw, pitch, roll, x_dot, y_dot, z_dot, yaw_rate, pitch_rate, roll_rate]^T
        :param u: action as motor currents
        :return x_dot: state time derivative
        '''
        m = sym if x.dtype == object else np

        #State
        euler_angles = x[3:6]
        linear_velocity = x[6:9]
        angular_velocity = x[9:]

        y,  p,  r = euler_angles.tolist()
        cr, cp, cy = m.cos(r), m.cos(p), m.cos(y)
        sr, sp, sy = m.sin(r), m.sin(p), m.sin(y)

        # Rotation matrix from body frame to world frame
        R_NB = np.array([[      cp*cy,        cy*sp*sr - sy*cr,    cy*sp*cr + sy*sr],
                        [       sy*cp,        sy*sp*sr + cy*cr,    sy*sp*cr - cy*sr],
                        [        -sp,               cp*sr,               cr*cp    ]])

        # =====================Linear Accel Calc===================== #

        # Compute thrust due to rotors in body frame
        uF_Bz = self.kF * u 
        Faero_B = np.array([0., 0., np.sum(uF_Bz)], dtype = x.dtype) 

        # Compute gravity and world frame
        Fgrav_N = np.array([0, 0, -self.m * 9.81], dtype = x.dtype)

        # Total forces
        Ftot_N = Fgrav_N + R_NB @ Faero_B

        linear_accel = Ftot_N / self.m

        # ====================Angular Accel Calc===================== #

        # Compute moments due to thrust 
        Mx_B = L * (uF_Bz[1] - uF_Bz[3])
        My_B = L * (uF_Bz[2] - uF_Bz[0])

        # Compute reaction moments due to actuating motors (Motor direction accounted for in summing)
        uTau_Bz = self.kM * u
        Mz_B = uTau_Bz[0] - uTau_Bz[1] + uTau_Bz[2] - uTau_Bz[3]

        # Total torque
        Tau_B = np.array([Mx_B, My_B, Mz_B], dtype = x.dtype)

        # Base angular velocity in world, expressed in base coordinates
        M = np.array([[1.,    0.,   -sp],
                        [0.,    cr,  sr*cp],
                        [0.,   -sr,  cr*cp]], dtype = x.dtype)

        w_BN_B = M @ angular_velocity
        # Compute angular acceleration in body frame, expressed in world coordinates
        alpha_NB_N = R_NB @ self.inv_MoI @ (Tau_B - np.cross(w_BN_B, self.MoI @ w_BN_B) )
        
        
        # Compute Amgular accleration in world coordinates
        Minv = np.array([[cy/cp,      sy/cp,     0.0],
                        [  -sy,         cy,      0.0],
                        [cy/cp*sp,   sy/cp*sp,   1.0]], dtype = x.dtype)
        
        pitch_rate = angular_velocity[1]
        yaw_rate = angular_velocity[0]
        
        Mdt =  np.array([[-cy*sp*pitch_rate - sy*cp*yaw_rate,   -cy*yaw_rate,    0.0,],
                         [-sy*sp*pitch_rate + cy*cp*yaw_rate,   -sy*yaw_rate,    0.0],
                         [           -cp*pitch_rate,                 0.0,        0.0]], dtype = x.dtype)
        
        angular_accel = Minv @ (alpha_NB_N - Mdt @ np.flip(angular_velocity))
        return np.hstack((linear_velocity, angular_velocity, linear_accel, np.flip(angular_accel)))     


    def discrete_dynamics(self, x0, u):
        x_next = RK4(self.continuous_dynamics, x0, u, dt = self.dt)
        return x_next
    
    def set_references(self, xref, uref):
        self.xref = xref
        self.uref = uref
        self.CostDerivatives.set_references(xref, uref)

    def _cost_stage(self, x: np.ndarray, u: np.ndarray, xref: np.ndarray, uref: np.ndarray):
        '''
        Computes cost due to state and action trajectory

        :param xtraj: state trajectory, ndarray of shape (N, nx)
        :param utraj: action trajectory, ndarray of shape (N-1, nu)
        '''

        xerr =  x - xref
        uerr = u - uref

        # Weighted errors
        weighted_state_errors = xerr @ self.Q  
        weighted_action = uerr@ self.R

        state_cost = np.sum(weighted_state_errors * xerr) 
        action_cost = np.sum(weighted_action * uerr)

        return 0.5*state_cost + 0.5* action_cost
    
    def _cost_final(self, xf: np.ndarray, xref: np.ndarray):
        '''
        Computes cost due to final state of state trajectory
        '''
        xerr = (xf - xref).T    
        weighted_state_error = xerr @ self.Qf

        return 0.5 * np.sum(weighted_state_error* xerr)
    
    def cost_trj(self, xtraj: np.ndarray, utraj: np.ndarray):
        '''
        Computes total cost of trajectory
        :param xtraj: state trajectory, ndarray of shape (num_time_steps, nx)
        :param utraj: action trajectory, ndarray of shape (num_time_steps-1, nu)

        '''
        J = 0
        xf = xtraj[-1, :]

        J += self._cost_stage(xtraj[:-1,:], utraj[:-1,:], self.xref, self.uref)
        J += self._cost_final(xf, self.xref)
        return J
    
    def stage(self, x, u):
        l_x, l_u, l_xx, l_ux, l_uu, f_x, f_u  = self.ModelAutoDiff.stage(x,u)
        return l_x, l_u, l_xx, l_ux, l_uu, f_x, f_u 
    
    def final(self, x):
        V_x, V_xx = self.ModelAutoDiff.final(x)
        return V_x, V_xx