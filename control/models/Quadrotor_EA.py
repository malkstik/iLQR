
from control.models.model import Model


from pydrake.all import (Diagram,
                         MultibodyPlant,
                         FirstOrderTaylorApproximation,
                         Simulator)

import numpy as np 
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))
from maths import autodiff

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
        f_x, f_u = self._linearize(self.xref, self.uref)
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
        f_x, f_u = self._linearize(self.xref, self.uref)
        return l_x, l_u, l_xx, l_ux, l_uu, f_x, f_u 
    
    def final(self, x):
        V_x, V_xx = self.CostDerivatives.final(x)
        return V_x, V_xx