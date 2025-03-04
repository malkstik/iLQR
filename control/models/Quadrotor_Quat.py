

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
from maths.forward_integration import RK4
import maths.quaternions as qmath
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
        self.ndx = 12

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

        ##---------------------------------Cost Functions---------------------------------##
        StateDiffJacobian = ca.MX.zeros(13, 12)
        StateDiffJacobian[:4, :3] = GetAttitudeJacobian(quat)
        StateDiffJacobian[4:, 3:] = ca.MX.eye(9)
        self._GetStateDiffJacobian = ca.Function('StateDiffJac', [x_sym], [StateDiffJacobian])

        xref = ca.MX.sym('xref', 13)  
        uref = ca.MX.sym('uref', 4)  

        state_err = StateDiffJacobian.T@(x_sym-xref)
        action_err = u_sym-uref
        stage_cost = 0.5*state_err.T@Q@state_err + 0.5*action_err.T@R@action_err
        terminal_cost = 0.5*state_err.T@Qf@state_err

        self.stage_cost_fun = ca.Function('stage_cost', [x_sym, u_sym, xref, uref], [stage_cost])
        self.terminal_cost_fun = ca.Function('terminal_cost', [x_sym, xref], [terminal_cost])

        l_x = ca.jacobian(stage_cost, x_sym)
        l_u = ca.jacobian(stage_cost, u_sym)

        l_xx = ca.jacobian(l_x, x_sym)
        l_ux = ca.jacobian(l_u, x_sym)        
        l_uu = ca.jacobian(l_u, u_sym)

        self.l_x_fun = ca.Function('l_x', [x_sym, u_sym, xref, uref], [l_x])
        self.l_u_fun = ca.Function('l_u', [x_sym, u_sym, xref, uref], [l_u])
        self.l_xx_fun = ca.Function('l_xx', [x_sym, u_sym, xref, uref], [l_xx])
        self.l_ux_fun = ca.Function('l_ux', [x_sym, u_sym, xref, uref], [l_ux])
        self.l_uu_fun = ca.Function('l_uu', [x_sym, u_sym, xref, uref], [l_uu])

        V_x_n = ca.jacobian(terminal_cost, x_sym)
        V_xx_n = ca.jacobian(V_x_n, x_sym)

        self.V_x_n_fun = ca.Function('V_x', [x_sym, xref], [V_x_n])
        self.V_xx_n_fun = ca.Function('V_xx', [x_sym, xref], [V_xx_n])


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
        return self.stage_cost_fun(x, u, xref, uref)

    def _cost_final(self, xf: np.ndarray, xref: np.ndarray):
        '''
        Computes cost due to final state of state trajectory
        '''
        return self.terminal_cost_fun(xf, xref)
    
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
    
    def CalcDifferentialState(self, state: np.ndarray, ref_state: np.ndarray):
        '''
        Computes differential state accounting for quaternion kinematics

        :param state: ndarray of shape (nx)
        :param ref_state: ndarray of shape (nx,)

        :return: ndarray of shape (nx-1)
        '''
        G = self._GetStateDiffJacobian(state)
        return G.T @ (state - ref_state) 


    def _linearize_discrete(self, ref_state, ref_action):
        A, B = self.A_discrete_fun(ref_state, ref_action), self.B_discrete_fun(ref_state, ref_action) 
        StateDiffJacobian = self._GetStateDiffJacobian(ref_state)

        discreteA = StateDiffJacobian.T @ A @ StateDiffJacobian
        discreteB = StateDiffJacobian.T @ B

        return discreteA, discreteB

    def _linearize_continuous(self,ref_state, ref_action):
        A, B = self.A_continuous_fun(ref_state, ref_action), self.B_continuous_fun(ref_state, ref_action) 
        StateDiffJacobian = self._GetStateDiffJacobian(ref_state)

        contA = StateDiffJacobian.T @ A @ StateDiffJacobian
        contB = StateDiffJacobian.T @ B        
        return contA, contB
    
    def stage(self, x, u):         
        f_x, f_u = self._linearize_discrete(x, u)

        l_x = self.l_x_fun(x, u, self.xref, self.uref)
        l_u = self.l_u_fun(x, u, self.xref, self.uref)
        l_xx = self.l_xx_fun(x, u, self.xref, self.uref)
        l_uu = self.l_uu_fun(x, u, self.xref, self.uref)
        l_ux = self.l_ux_fun(x, u, self.xref, self.uref)

        return l_x, l_u, l_xx, l_ux, l_uu, f_x, f_u 
    
    def final(self, x):
        G = self._GetStateDiffJacobian(x)

        V_xx = G.T @ self.V_xx_n_fun(x, self.xref) @ G
        V_x = self.V_x_n_fun(x, self.xref) @ G

        return V_x.T, V_xx
    
    def Q_terms(self, state, l_x, l_u, l_xx, l_ux, l_uu, f_x, f_u, V_x, V_xx):
        G = self._GetStateDiffJacobian(state)

        Q_x = (l_x @ G).T + f_x.T @ V_x
        Q_u = l_u.T + f_u.T @ V_x 
        Q_xx = G.T @ l_xx @ G + f_x.T @ V_xx @ f_x
        Q_ux = l_ux @ G + f_u.T @ V_xx @ f_x
        Q_uu = l_uu + f_u.T @ V_xx @ f_u
        return Q_x, Q_u, Q_xx, Q_ux, Q_uu
    

class DrakeQuadrotorQuatModel(Model):
    '''
    Quadrotor model class defining defining dynamics of quadrotor with states 
    [qw, qx, qy, qz, x, y, z, wx, wy, wz, vx, vy, vz]^T

    and stage cost
    (x-xref).T @ G(q).T @ Q @ (x-xref)@G(q) + u.T@R@u
    '''
    def __init__(self,
                 quadrotor,
                 multibody_plant,
                 Q: np.ndarray,
                 Qf: np.ndarray,
                 R: np.ndarray,
                 dt: int = 0.01,
                 N: int = 20,
                 integration: str = "RK4",
                 **kwargs):

        self.nx = 13
        self.nu = 4
        self.ndx = 12

        self.Q = Q
        self.Qf = Qf
        self.R = R
        self.dt = dt
        self.N = N

        self.xref = np.zeros(self.nx, dtype = np.float64)
        self.uref = np.zeros(self.nu, dtype = np.float64)

        self.quadrotor = quadrotor
        self.plant = multibody_plant


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
        return self.stage_cost_fun(x, u, xref, uref)

    def _cost_final(self, xf: np.ndarray, xref: np.ndarray):
        '''
        Computes cost due to final state of state trajectory
        '''
        return self.terminal_cost_fun(xf, xref)
    
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
    
    def CalcDifferentialState(self, state: np.ndarray, ref_state: np.ndarray):
        '''
        Computes differential state accounting for quaternion kinematics

        :param state: ndarray of shape (nx)
        :param ref_state: ndarray of shape (nx,)

        :return: ndarray of shape (nx-1)
        '''
        G = self._GetStateDiffJacobian(state)
        return G.T @ (state - ref_state) 

    def _GetStateDiffJacobian(ref_state):
        qo = ref_state[:4]
        StateDiffJacobian = np.zeros((13, 12), dtype = q0.dtype)
        StateDiffJacobian[:4, :3] = qmath.GetAttitudeJacobian(q0)
        StateDiffJacobian[4:, 3:] = np.eye(9)

        return StateDiffJacobian


    def _linearize_discrete(self, ref_state, ref_action):
        A, B = self.A_discrete_fun(ref_state, ref_action), self.B_discrete_fun(ref_state, ref_action) 
        StateDiffJacobian = self._GetStateDiffJacobian(ref_state)

        discreteA = StateDiffJacobian.T @ A @ StateDiffJacobian
        discreteB = StateDiffJacobian.T @ B

        return discreteA, discreteB

    def _linearize_continuous(self,ref_state, ref_action):
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
        
        q0 = context.get_continuous_state_vector().CopyToVector()[:4].reshape((4,1))
        StateDiffJacobian = self._GetStateDiffJacobian(q0)

        A, B = sys.A(), sys.B()    
        A = StateDiffJacobian.T @ sys.A() @ StateDiffJacobian
        B = StateDiffJacobian.T @ sys.B()

        return A, B


    def _linearize_discrete(self, ref_state, ref_action):

        '''
        Perform first order taylor approximation on system dynamics andd return discrete form
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
        
        discrete_sys = sys.
        q0 = context.get_continuous_state_vector().CopyToVector()[:4].reshape((4,1))
        StateDiffJacobian = self._GetStateDiffJacobian(q0)

        A, B = sys.A(), sys.B()    
        A = StateDiffJacobian.T @ sys.A() @ StateDiffJacobian
        B = StateDiffJacobian.T @ sys.B()

        return A, B
    
    def stage(self, x, u):         
        f_x, f_u = self._linearize_discrete(x, u)

        l_x = self.l_x_fun(x, u, self.xref, self.uref)
        l_u = self.l_u_fun(x, u, self.xref, self.uref)
        l_xx = self.l_xx_fun(x, u, self.xref, self.uref)
        l_uu = self.l_uu_fun(x, u, self.xref, self.uref)
        l_ux = self.l_ux_fun(x, u, self.xref, self.uref)

        return l_x, l_u, l_xx, l_ux, l_uu, f_x, f_u 
    
    def final(self, x):
        G = self._GetStateDiffJacobian(x)

        V_xx = G.T @ self.V_xx_n_fun(x, self.xref) @ G
        V_x = self.V_x_n_fun(x, self.xref) @ G

        return V_x.T, V_xx
    
    def Q_terms(self, state, l_x, l_u, l_xx, l_ux, l_uu, f_x, f_u, V_x, V_xx):
        G = self._GetStateDiffJacobian(state)

        Q_x = (l_x @ G).T + f_x.T @ V_x
        Q_u = l_u.T + f_u.T @ V_x 
        Q_xx = G.T @ l_xx @ G + f_x.T @ V_xx @ f_x
        Q_ux = l_ux @ G + f_u.T @ V_xx @ f_x
        Q_uu = l_uu + f_u.T @ V_xx @ f_u
        return Q_x, Q_u, Q_xx, Q_ux, Q_uu