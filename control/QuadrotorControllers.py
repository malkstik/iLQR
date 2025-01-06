from pydrake.all import (LeafSystem,
                         BasicVector,
                         Diagram,
                         MultibodyPlant,
                         FirstOrderTaylorApproximation,
                         LinearQuadraticRegulator,
                         Context,
                         Simulator)

import numpy as np
import math 
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))
import maths.quaternions as qmath
from maths import autodiff
from maths import linalg 

from time import time

class QuadrotorController(LeafSystem):
    """Base controller class for quadrotor using quaternion floating base
    """
    def __init__(self, quadrotor: Diagram, multibody_plant: MultibodyPlant):
            # 13 inputs (quadrotor state), 4 motor current outputs.
            LeafSystem.__init__(self)
            self.quadrotor = quadrotor
            self.plant = multibody_plant

            self.mass = multibody_plant.CalcTotalMass(multibody_plant.GetMyContextFromRoot(quadrotor.CreateDefaultContext()))


            self.DeclareVectorInputPort("current_state", 13)
            self.DeclareVectorInputPort("goal_state", 13)
            self.DeclareVectorInputPort("ref_action", 4)
            self.DeclareVectorOutputPort("control_output", 4, self.DoCalcVectorOutput)


    def DoCalcVectorOutput(self, context: Context, motor_current: np.ndarray):
        return NotImplemented

    def _GetStateDiffJacobian(self, q0: np.ndarray):
        StateDiffJacobian = np.zeros((13, 12), dtype = q0.dtype)
        StateDiffJacobian[:4, :3] = qmath.GetAttitudeJacobian(q0)
        StateDiffJacobian[4:, 3:] = np.eye(9)

        return StateDiffJacobian

    def _Linearize(self, ref_state: np.ndarray, ref_action: np.ndarray, ReduceState: bool = True):
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
        if ReduceState:        
            A = StateDiffJacobian.T @ sys.A() @ StateDiffJacobian
            B = StateDiffJacobian.T @ sys.B()

        return A, B, StateDiffJacobian
    

    def Rollout(self, x0: np.ndarray, U: np.ndarray, dt: float, timeit: bool = False):
        """
        Simulates the trajectory of states (rollout) for given initial state and control sequence.

        :param x0: initial state, ndarray of shape (nx,)
        :param U: sequence of actions, ndarray of shape (num_time_steps-1, nu)
        :param dt: 

        :return x: sequence of states from rollout, ndarray of shape (num_time_steps, nx)
        """

        X = [x0.copy().tolist()]


        simulator = Simulator(self.quadrotor)
        simulator.Initialize()
        simulator_context = simulator.get_mutable_context()
        simulator_context.SetContinuousState(x0)

        for u in U:
            # Set input and simulate a small time step

            self.quadrotor.get_input_port().FixValue(simulator_context, u)
            
            # Use the plant's dynamics (or a numerical integrator) to step forward in time
            # This should ideally be a single step of the plant's integrator
            # simulator = Simulator(self.plant, context)
            sim_time = simulator_context.get_time()

            simulator.AdvanceTo(sim_time + dt)
            x_next = simulator_context.get_continuous_state_vector().CopyToVector()

            X.append(x_next)

            # print(f"time: {sim_time + dt} \nx_next: {x_next} \nu: {u} \n")

        #normalize quaternions
        xtraj = np.array(X)
        xtraj[:, :4] *= np.reciprocal(np.linalg.norm(xtraj[:, :4], axis= 1)).reshape((-1, 1))

        return xtraj

    
    def CreateDefaultContext(self):
        return self.quadrotor.CreateDefaultContext()    
    
    def _ComputeDifferentialState(self, state: np.ndarray, ref_state: np.ndarray):
        '''
        Computes differential state accounting for quaternion kinematics

        :param state: ndarray of shape (batch, nx)
        :param ref_state: ndarray of shape (nx,)

        :return: ndarray of shape (batch, nx-1)
        '''

        q_ref = ref_state[:4]
        q = state[:,:4]

        quaternion_error = qmath.GetLeftMatrix(q_ref).T @ q.T

        if(np.any(quaternion_error[0]) <= 0.001):
            print(f"q_ref: {q_ref}\n q: {q}")

        differential_quaternion = qmath.QuaternionToParam(quaternion_error).T


        return np.hstack((differential_quaternion, state[:, 4:] - ref_state[4:]))

class QuadrotorLQR(QuadrotorController):
    """Define LQR controller for quadrotor using quaternion floating base
    """

    def __init__(self, quadrotor: Diagram, multibody_plant: MultibodyPlant, Q: np.ndarray, R: np.ndarray):
        # 13 inputs (quadrotor state), 4 motor current outputs.
        super().__init__(quadrotor, multibody_plant)
        self.quadrotor = quadrotor
        self.plant = multibody_plant
        self.Q = Q
        self.R = R

        self.ref_state = np.array([1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.])
        self.ref_action = np.zeros(4)

        self._SetReferencePoint(self.ref_state, self.ref_action)

    def DoCalcVectorOutput(self, context: Context, motor_current: BasicVector):
        current_state = self.get_input_port(0).Eval(context)
        goal_state = self.get_input_port(1).Eval(context)
        ref_action = self.get_input_port(2).Eval(context)
        self._SetReferencePoint(goal_state, ref_action)

        differential_quadrotor_state = self._ComputeDifferentialState(current_state.reshape((1, -1)), self.ref_state)
        # motor_current[:] = self.ref_action - self.K @ differential_quadrotor_state   

        motor_current.SetFromVector((self.ref_action - differential_quadrotor_state @ self.K.T).squeeze())
        
    def _SetReferencePoint(self, ref_state: np.ndarray, ref_action: np.ndarray):
        '''
        Updates reference state and action then recomputes linearization and optimal feedback gain
        :param ref_state: reference state consisting of 
        [qw, qx, qy, qz, x, y, z, wx, wy, wz, vx, vy, vz]^T
        
        :param ref_action: reference action consisting of 
        [motor_current_i for i in [0, 1, 2, 3]]
        '''
        if (self.ref_state == ref_state).all() and (self.ref_action == ref_action).all():
            return

        self.ref_state = ref_state
        self.ref_action = ref_action

        A, B, _ = self._Linearize(ref_state, ref_action)
        self.K, _ = LinearQuadraticRegulator(A, B, self.Q, self.R)        

class QuadrotoriLQR(QuadrotorController):
    def __init__(self, quadrotor: Diagram, 
                 multibody_plant: MultibodyPlant, 
                 Q: np.ndarray, 
                 R: np.ndarray,
                 Qf: np.ndarray,
                 Tf: float,
                 dt: float,
                 max_iters: int = 100,
                 init_regu: float = 0.05,
                 **kwargs):
        
        super().__init__(quadrotor, multibody_plant)
        self.Q: np.ndarray = Q
        self.R: np.ndarray = R
        self.Qf: np.ndarray = Qf
        self.Tf: float = Tf
        self.dt: float = dt
        self.max_iters: int = max_iters
        self.init_regu: float = init_regu

        self.nx = 13 #full state size
        self.nx_diff = 12 #differential state size
        self.nu = 4
        self.num_time_steps: int = math.floor(Tf/dt) + 1

        self.xtraj: np.ndarray = np.zeros((self.num_time_steps, self.nx))
        self.utraj: np.ndarray = np.zeros((self.num_time_steps-1, self.nu))
        self.xgoal = np.zeros(self.nx)
        self.ugoal = np.zeros(self.nu)
        self.CostDerivatives = autodiff.CostDerivatives(self._stage_cost,
                                                        self._terminal_cost,
                                                        self.xgoal,
                                                        self.ugoal,
                                                        self.nx,
                                                        self.nu
                                                        ) 

    def InitTraj(self, x0: np.ndarray, utraj: np.ndarray):
         
        self.xtraj[:] = np.kron(np.ones(self.num_time_steps, 1), x0)
        self.utraj[:] = utraj

    def DoCalcVectorOutput(self, context: Context, motor_current: BasicVector):
        current_state = self.get_input_port(0).Eval(context)
        goal_state = self.get_input_port(1).Eval(context)

        if not self.xtraj.any() and not self.utraj.any():
            self.xtraj[:] = np.kron(np.ones((self.num_time_steps, 1)), current_state) #Maybe consider only passing x0 since this gets overwritten in self._control
            # self.utraj[:] = 1.95* np.random.randn(self.num_time_steps-1, self.nu)
            self.utraj[:] = 1.95* np.ones((self.num_time_steps-1, self.nu)) #Normally distribute about a hover @ identity rotation

        self.xtraj, self.utraj = self._control(current_state, goal_state, self.xtraj, self.utraj)
        motor_current.SetFromVector(self.utraj[0])

    def _control(self, x0, xgoal, xtraj, utraj):
        self.xgoal = xgoal
        if (self.CostDerivatives.xref is not self.xgoal):
            self.CostDerivatives.set_references(self.xgoal, self.ugoal)

        #Initial Rollout
        xtraj = self.Rollout(x0, utraj, self.dt)

        J = self._cost(xtraj, utraj)

        self.iter = 0
        gradV = np.zeros((self.num_time_steps, self.nx_diff))                   #gradients
        hessV = np.zeros((self.num_time_steps, self.nx_diff, self.nx_diff))     #hessians 
        d = np.ones((self.num_time_steps-1, self.nu))                           #feedforward
        K = np.zeros((self.num_time_steps-1, self.nu, self.nx_diff))            #feedback


        while np.max(np.linalg.norm(d, axis = 0)) > 0.1 and self.iter < self.max_iters:
            self.iter += 1
            deltaJ = 0.0                                                            #change to trajectory cost             
            
            deltaJ, K, d[:] = self._backward_pass(xtraj, utraj, gradV, hessV, d, K)
            # Forward rollout with line search
            alpha = 1.0
            xtraj[:], utraj[:], Jn = self._forward_rollout(xtraj, utraj, d, K, alpha)
            
            while (np.isnan(Jn) or Jn > (J - 1e-2 * alpha * deltaJ)) and alpha < 0.03125:
                print(f"Jn - J: {Jn - J}\nReq Improvement: {-1e-2 * alpha * deltaJ}\n\n")
                alpha *= 0.5
                xtraj, utraj, Jn = self._forward_rollout(xtraj, utraj, d, K, alpha)
            J = Jn

        return xtraj, utraj
    
    
    def _backward_pass(self, xtraj, utraj, gradV, hessV, d, K):
        deltaJ = 0

        xN = np.copy(xtraj[-1, :])
        G = self._GetStateDiffJacobian(xN[:4])

        FinalGradV, FinalHessV = self.CostDerivatives.final(xN)

        gradV[-1, :] = G.T @ FinalGradV 
        hessV[-1, :, :] = G.T @ FinalHessV @ G
        for k in range(self.num_time_steps-1)[::-1]:
            xk, uk = np.copy(xtraj[k, :]), np.copy(utraj[k, :])
            hessV_k1 = hessV[k+1,:,:]
            gradV_k1 = gradV[k+1, :]

            # Linearization
            A, B, G = self._Linearize(xk, uk)
            controllable = linalg.is_controllable(A, B)
            # if not controllable:
            #     raise RuntimeWarning(f'''Uncontrollable linearization on backward pass {k} of iteration {self.iter}
            #                          xk:
            #                          {xk}
            #                          uk: 
            #                          {uk}
            #                          A:
            #                          {A}
            #                          B:
            #                          {B}''')

            # Compute Cost Derivatives
            l_x, l_u, l_xx, l_ux, l_uu = self.CostDerivatives.stage(xk, uk)

            # Determine action value derivatives
            Q_xx = G.T @ l_xx @ G + A.T @ hessV_k1 @ A
            Q_ux = l_ux @ G + B.T @ hessV_k1 @ A
            Q_xu = Q_ux.T
            Q_uu = l_uu + B.T @ hessV_k1 @ B 

            Q_x = G.T @ l_x + A.T @ gradV_k1
            Q_u = l_u + B.T @ gradV_k1


            rho = self.init_regu
            while linalg.is_singular(Q_uu):
                Q_uu += rho*np.eye(self.nu)
                rho += 0.01
                # raise RuntimeWarning(f"Singular l_uu on backward pass {k} of iteration {self.iter}")

            Q_uu_inv = linalg.qr_inverse(Q_uu)

            # Feedforward term
            d[k, :] = -Q_uu_inv @ Q_u
            d_k = d[k, :]
            # Feedback term
            K[k, :, :] = -Q_uu_inv @ Q_ux
            K_k = K[k, :, :]
            #update value function gradient and hessian

            hessV[k, :, :] = Q_xx + K_k.T@Q_uu@K_k + K_k.T@Q_ux + Q_xu@K_k

            gradV[k, :] = Q_x + K_k.T@Q_uu@d_k + K_k.T@Q_u + Q_xu@d_k
            
            deltaJ += -d_k.T@Q_u - 0.5*d_k.T@Q_uu@d_k

        return deltaJ, K, d
    
    def _forward_rollout(self, xtraj, utraj, d, K, alpha):

        xn = np.zeros((self.num_time_steps, self.nx))
        xn[0, :] = xtraj[0, :] #Set initial states to be equal
        un = np.zeros((self.num_time_steps-1, self.nu))

        for k in range(self.num_time_steps-1):
            # quat_diff = qmath.H.T @ qmath.GetLeftMatrix(qmath.GetConjugate(xtraj[k,:4]))@ xn[k,:4]

            # state_diff = np.hstack((quat_diff,
            #                        xn[k,4:] - xtraj[k,4:]))

            # un[k, :] = utraj[k, :] + alpha*d[k, :] + K[k, :, :] @ state_diff         

            un[k, :] = utraj[k, :] + alpha*d[k, :] + (K[k, :, :] @ self._ComputeDifferentialState(np.expand_dims(xn[k, :],0),
                                                                                                xtraj[k, :]).T).squeeze()            
            xn[k + 1, :] = self.Rollout(xn[k, :], np.expand_dims(un[k, :], 0), self.dt)[1,:]
        Jn = self._cost(xn, un)

        return xn, un, Jn

    def _stage_cost(self, x: np.ndarray, u: np.ndarray, xref: np.ndarray, uref: np.ndarray):
        '''
        Computes cost due to state and action trajectory

        :param xtraj: state trajectory, ndarray of shape (num_time_steps-1, nx)
        :param utraj: action trajectory, ndarray of shape (num_time_steps-1, nu)
        '''
        # xerr = self._ComputeDifferentialState(x, self.xgoal)
        G = self._GetStateDiffJacobian(x[:4])
        xerr = G.T @ (x - xref).T

        uerr = u - uref
        # Weighted errors
        weighted_state_errors = xerr @ self.Q  # Shape (15, 12)
        weighted_action = uerr@ self.R

        state_cost = np.sum(weighted_state_errors * xerr) 
        action_cost = np.sum(weighted_action * uerr)

        return 0.5*state_cost + 0.5* action_cost

    def _terminal_cost(self, xf: np.ndarray, xref: np.ndarray):
        '''
        Computes cost due to final state of state trajectory
        '''
        # xerr = self._ComputeDifferentialState(xf, self.xgoal)
        G = self._GetStateDiffJacobian(xf[:4])
        xerr = G.T @ (xf - xref).T    
        weighted_state_error = xerr @ self.Qf

        return 0.5 * np.sum(weighted_state_error* xerr)


    def _cost(self, xtraj: np.ndarray, utraj: np.ndarray):
        '''
        Computes total cost of trajectory
        :param xtraj: state trajectory, ndarray of shape (num_time_steps, nx)
        :param utraj: action trajectory, ndarray of shape (num_time_steps-1, nu)

        '''
        J = 0
        xf = xtraj[-1, :]

        J += self._stage_cost(xtraj[:-1,:], utraj[:-1,:], self.xgoal, self.ugoal)
        J += self._terminal_cost(xf, self.xgoal)
        return J

    def _full_hessian(self, Gxx: np.ndarray, Gxu: np.ndarray, Gux: np.ndarray, Guu: np.ndarray)-> np.ndarray:
        '''
        Combines hessian parts into one
        '''
        G = np.vstack( 
            (np.hstack((Gxx, Gxu)),
             np.hstack((Gux, Guu)))
            )
        return G
    
class QuadrotorController_EA(LeafSystem):
    """controller base class for quadrotor using euler (RPY) angle floating base
    """
    def __init__(self, quadrotor: Diagram, multibody_plant: MultibodyPlant):
                # 12 inputs (quadrotor state), 4 motor current outputs.
                LeafSystem.__init__(self)
                self.quadrotor = quadrotor
                self.plant = multibody_plant

                self.mass = multibody_plant.CalcTotalMass(multibody_plant.GetMyContextFromRoot(quadrotor.CreateDefaultContext()))


                self.DeclareVectorInputPort("current_state", 12)
                self.DeclareVectorInputPort("goal_state", 12)
                self.DeclareVectorInputPort("ref_action", 4)
                self.DeclareVectorOutputPort("control_output", 4, self.DoCalcVectorOutput) 

    def _Linearize(self, ref_state: np.ndarray, ref_action: np.ndarray, ReduceState: bool = True):
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
    

    def Rollout(self, x0: np.ndarray, U: np.ndarray, dt: float, timeit: bool = False):
        """
        Simulates the trajectory of states (rollout) for given initial state and control sequence.

        :param x0: initial state, ndarray of shape (nx,)
        :param U: sequence of actions, ndarray of shape (num_time_steps-1, nu)
        :param dt: 

        :return x: sequence of states from rollout, ndarray of shape (num_time_steps, nx)
        """

        X = [x0.copy().tolist()]


        simulator = Simulator(self.quadrotor)
        simulator.Initialize()
        simulator_context = simulator.get_mutable_context()
        simulator_context.SetContinuousState(x0)

        for u in U:
            # Set input and simulate a small time step

            self.quadrotor.get_input_port().FixValue(simulator_context, u)
            
            # Use the plant's dynamics (or a numerical integrator) to step forward in time
            # This should ideally be a single step of the plant's integrator
            # simulator = Simulator(self.plant, context)
            sim_time = simulator_context.get_time()

            simulator.AdvanceTo(sim_time + dt)
            x_next = simulator_context.get_continuous_state_vector().CopyToVector()

            X.append(x_next)

            # print(f"time: {sim_time + dt} \nx_next: {x_next} \nu: {u} \n")

        return np.array(X)

    
    def CreateDefaultContext(self):
        return self.quadrotor.CreateDefaultContext()    

    def InitTraj(self, x0: np.ndarray, utraj: np.ndarray):
         
        self.xtraj[:] = np.kron(np.ones(self.num_time_steps, 1), x0)
        self.utraj[:] = utraj

    def DoCalcVectorOutput(self, context: Context, motor_current: BasicVector):
        raise NotImplementedError

class QuadrotorLQR_EA(QuadrotorController_EA):
    """Define LQR controller for quadrotor using RPY floating base
    """
    def __init__(self, quadrotor: Diagram, multibody_plant: MultibodyPlant, Q: np.ndarray, R: np.ndarray):
        # 12 inputs (quadrotor state), 4 motor current outputs.
        super().__init__(quadrotor, multibody_plant)
        self.quadrotor = quadrotor
        self.plant = multibody_plant
        self.Q = Q
        self.R = R

        self.ref_state = np.array([0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.ref_action = 1.91 * np.zeros(1)

        self._SetReferencePoint(self.ref_state, self.ref_action)

    def DoCalcVectorOutput(self, context: Context, motor_current: BasicVector):
        current_state = self.get_input_port(0).Eval(context)
        goal_state = self.get_input_port(1).Eval(context)
        ref_action = self.get_input_port(2).Eval(context)
        self._SetReferencePoint(goal_state, ref_action)

        state_diff = current_state - goal_state
        motor_current.SetFromVector((self.ref_action - state_diff @ self.K.T).squeeze())
        
    def _SetReferencePoint(self, ref_state: np.ndarray, ref_action: np.ndarray):
        '''
        Updates reference state and action then recomputes linearization and optimal feedback gain
        :param ref_state: reference state consisting of 
        [qw, qx, qy, qz, x, y, z, wx, wy, wz, vx, vy, vz]^T
        
        :param ref_action: reference action consisting of 
        [motor_current_i for i in [0, 1, 2, 3]]
        '''
        if (self.ref_state == ref_state).all() and (self.ref_action == ref_action).all():
            return

        self.ref_state = ref_state
        self.ref_action = ref_action

        A, B = self._Linearize(ref_state, ref_action)
        self.K, _ = LinearQuadraticRegulator(A, B, self.Q, self.R)     

class QuadrotoriLQR_EA(QuadrotorController_EA):
    """iLQR controller for quadrotor using euler (RPY) angle floating base
    """
    def __init__(self, quadrotor: Diagram, 
                 multibody_plant: MultibodyPlant, 
                 Q: np.ndarray, 
                 R: np.ndarray,
                 Qf: np.ndarray,
                 Tf: float,
                 dt: float,
                 max_iters: int = 100,
                 init_regu: float = 100,
                 min_regu: float = 0.01,
                 max_regu: float = 10000,
                 **kwargs):
            # 13 inputs (quadrotor state), 4 motor current outputs.
            LeafSystem.__init__(self)
            self.quadrotor = quadrotor
            self.plant = multibody_plant

            self.mass = multibody_plant.CalcTotalMass(multibody_plant.GetMyContextFromRoot(quadrotor.CreateDefaultContext()))

            self.DeclareVectorInputPort("current_state", 12)
            self.DeclareVectorInputPort("goal_state", 12)
            self.DeclareVectorInputPort("ref_action", 4)
            self.DeclareVectorOutputPort("control_output", 4, self.DoCalcVectorOutput)

            self.Q: np.ndarray = Q
            self.R: np.ndarray = R
            self.Qf: np.ndarray = Qf
            self.Tf: float = Tf
            self.dt: float = dt
            self.max_iters: int = max_iters
            self.init_regu: float = init_regu
            self.min_regu = min_regu
            self.max_regu = max_regu

            self.nx = 12 #full state size
            self.nu = 4
            self.num_time_steps: int = math.floor(Tf/dt) + 1

            self.xtraj: np.ndarray = np.zeros((self.num_time_steps, self.nx))
            self.utraj: np.ndarray = np.zeros((self.num_time_steps-1, self.nu))
            self.xgoal = np.zeros(self.nx)
            self.ugoal = np.zeros(self.nu)
            self.CostDerivatives = autodiff.CostDerivatives(self._stage_cost,
                                                            self._terminal_cost,
                                                            self.xgoal,
                                                            self.ugoal,
                                                            self.nx,
                                                            self.nu
                                                            )             
    def DoCalcVectorOutput(self, context: Context, motor_current: BasicVector):
        current_state = self.get_input_port(0).Eval(context)
        goal_state = self.get_input_port(1).Eval(context)

        if not self.xtraj.any() and not self.utraj.any():
            self.utraj[:] = 1.95* np.ones((self.num_time_steps-1, self.nu)) #Normally distribute about a hover @ identity rotation

        self.xtraj, self.utraj = self._control(current_state, goal_state, self.utraj)
        motor_current.SetFromVector(self.utraj[0])

    # def _control(self, x0, xgoal, xtraj, utraj):
    #     self.xgoal = xgoal
    #     if (self.CostDerivatives.xref is not self.xgoal or self.CostDerivatives.uref is not self.ugoal):
    #         self.CostDerivatives.set_references(self.xgoal, self.ugoal)

    #     #Initial Rollout
    #     xtraj = self.Rollout(x0, utraj, self.dt)

    #     J = self._cost(xtraj, utraj)

    #     self.iter = 0
    #     d = np.ones((self.num_time_steps-1, self.nu))                       #feedforward
    #     K = np.zeros((self.num_time_steps-1, self.nu, self.nx))             #feedback
    #     regu = self.init_regu

        
    #     Jprev = 0
    #     while J - Jprev > 0.5:
    #         print(J - Jprev)
    #         Jprev = J
    #         print(self.iter)
    #         self.iter += 1
    #         deltaJ = 0.0                                                            #change to trajectory cost             
            
    #         deltaJ, K[:], d[:] = self._backward_pass(xtraj, utraj, d, K, regu)
    #         # Forward rollout with line search
    #         alpha = 1.0
    #         # xtraj[:], utraj[:], Jn = self._forward_rollout(xtraj, utraj, d, K, alpha)
    #         xtraj_new, utraj_new, Jn = self._forward_rollout(xtraj, utraj, d, K, alpha)
            
    #         while (np.isnan(Jn) or Jn > (J - 1e-2 * alpha * deltaJ)) and 1e-2 * alpha * deltaJ > 0.001:
    #             print(f"J - Jn: {J - Jn}\nReq Improvement: {1e-2 * alpha * deltaJ}\n\n")
    #             alpha *= 0.5
    #             xtraj_new, utraj_new, Jn = self._forward_rollout(xtraj, utraj, d, K, alpha)
            
    #         xtraj[:] = xtraj_new
    #         utraj[:] = utraj_new
    #         J = Jn

    #     return xtraj, utraj
        
    def _control(self, x0, xgoal, utraj):

        self.xgoal = xgoal
        if (self.CostDerivatives.xref is not self.xgoal or self.CostDerivatives.uref is not self.ugoal):
            self.CostDerivatives.set_references(self.xgoal, self.ugoal)

        #Initial Rollout
        xtraj = self.Rollout(x0, utraj, self.dt)

        J = self._cost(xtraj, utraj)

        self.iter = 0
        d = np.ones((self.num_time_steps-1, self.nu))                       #feedforward
        K = np.zeros((self.num_time_steps-1, self.nu, self.nx))             #feedback

        regu = self.init_regu
        for it in range(self.max_iters):
            self.iter += 1
            deltaJ = 0.0                                                            #change to trajectory cost             
            alpha = 0.5

            # start = time()
            deltaJ, K[:], d[:] = self._backward_pass(xtraj, utraj, d, K, regu)
            # print(f"Backward pass {self.iter} took {time()- start} s")
            
            # start = time()
            xtraj_new, utraj_new, Jn = self._forward_rollout(xtraj, utraj, d, K, alpha)
            # print(f"Rollout {self.iter} took {time()- start} s")
            # print()

            cost_redu = J - Jn

            if cost_redu > 0:
                xtraj[:] = xtraj_new
                utraj[:] = utraj_new
                regu *= 0.7
            else:
                regu *= 2.0
            regu = min(max(regu, self.min_regu), self.max_regu)

            if deltaJ <= 1e-6:
                print("early termination")
                break
        return xtraj, utraj
    
    def _backward_pass(self, xtraj, utraj, d, K, regu):
        deltaJ = 0

        xN = np.copy(xtraj[-1, :])
        V_x, V_xx = self.CostDerivatives.final(xN)
        for k in range(self.num_time_steps-1)[::-1]:
            xk, uk = np.copy(xtraj[k, :]), np.copy(utraj[k, :])
            # Linearization
            A, B = self._Linearize(xk, uk)
            l_x, l_u, l_xx, l_ux, l_uu = self.CostDerivatives.stage(xk, uk)
            
            Q_x, Q_u, Q_xx, Q_ux, Q_uu = self._Q_terms(l_x, l_u, l_xx, l_ux, l_uu, A, B, V_x, V_xx)


            Q_uu += regu*np.eye(self.nu)
            # Feedforward term

            d_k, K_k = self._gains(Q_uu, Q_u, Q_ux)
            d[k, :] = d_k
            # Feedback term
            K[k, :, :] = K_k
            #update value function gradient and hessian
            V_x, V_xx = self._V_terms(Q_x, Q_u, Q_xx, Q_ux, Q_uu, K_k, d_k)
            deltaJ += self._expected_cost_reduction(Q_u, Q_uu, d_k)
    
        return deltaJ, K, d
    
    def _forward_rollout(self, xtraj, utraj, d, K, alpha):

        xn = np.zeros((self.num_time_steps, self.nx))
        xn[0, :] = xtraj[0, :] #Set initial states to be equal
        un = np.zeros((self.num_time_steps-1, self.nu))

        for k in range(self.num_time_steps-1):
            un[k, :] = utraj[k, :] + alpha*d[k, :] + (K[k, :, :] @ (xn[k, :] - xtraj[k, :]).T).squeeze()            
            xn[k + 1, :] = self.Rollout(xn[k, :], np.expand_dims(un[k, :], 0), self.dt)[1,:]
        Jn = self._cost(xn, un)

        return xn, un, Jn

    def _stage_cost(self, x: np.ndarray, u: np.ndarray, xref: np.ndarray, uref: np.ndarray):
        '''
        Computes cost due to state and action trajectory

        :param xtraj: state trajectory, ndarray of shape (num_time_steps-1, nx)
        :param utraj: action trajectory, ndarray of shape (num_time_steps-1, nu)
        '''
        # xerr = self._ComputeDifferentialState(x, self.xgoal)
        xerr =  x - xref
        uerr = u - uref

        # Weighted errors
        weighted_state_errors = xerr @ self.Q  
        weighted_action = uerr@ self.R

        state_cost = np.sum(weighted_state_errors * xerr) 
        action_cost = np.sum(weighted_action * uerr)

        return 0.5*state_cost + 0.5* action_cost

    def _terminal_cost(self, xf: np.ndarray, xref: np.ndarray):
        '''
        Computes cost due to final state of state trajectory
        '''
        # xerr = self._ComputeDifferentialState(xf, self.xgoal)
        xerr = (xf - xref).T    
        weighted_state_error = xerr @ self.Qf

        return 0.5 * np.sum(weighted_state_error* xerr)

    def _cost(self, xtraj: np.ndarray, utraj: np.ndarray):
        '''
        Computes total cost of trajectory
        :param xtraj: state trajectory, ndarray of shape (num_time_steps, nx)
        :param utraj: action trajectory, ndarray of shape (num_time_steps-1, nu)

        '''
        J = 0
        xf = xtraj[-1, :]

        # for k in range(self.num_time_steps-1):
        #     J += self._stage_cost(xtraj[k,:], utraj[k,:], self.xgoal, self.ugoal)

        J += self._stage_cost(xtraj[:-1,:], utraj[:-1,:], self.xgoal, self.ugoal)
        J += self._terminal_cost(xf, self.xgoal)
        return J

    def _Q_terms(self, l_x, l_u, l_xx, l_ux, l_uu, f_x, f_u, V_x, V_xx):
        Q_xx = l_xx + f_x.T @ V_xx @ f_x
        Q_x = l_x + f_x.T @ V_x
        Q_u = l_u + f_u.T @ V_x

        Q_ux = l_ux + f_u.T @ V_xx @ f_x
        Q_uu = l_uu + f_u.T @ V_xx @ f_u
        return Q_x, Q_u, Q_xx, Q_ux, Q_uu    

    def _gains(self, Q_uu, Q_u, Q_ux):
        Q_uu_inv = np.linalg.inv(Q_uu)
        k = -Q_uu_inv @ Q_u
        K = -Q_uu_inv@ Q_ux
        return k, K

    def _V_terms(self, Q_x, Q_u, Q_xx, Q_ux, Q_uu, K, k):

        V_xx = Q_xx + K.T @ Q_uu @ K + K.T @ Q_ux + Q_ux.T @ K
        V_x = Q_x + K.T @ Q_uu @ k + K.T @ Q_u + Q_ux.T @ k
        return V_x, V_xx

    def _expected_cost_reduction(self, Q_u, Q_uu, k):
        return -Q_u.T.dot(k) - 0.5 * k.T.dot(Q_uu.dot(k))

if __name__ == "__main__":
    print("run")