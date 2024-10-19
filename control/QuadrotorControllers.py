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
from maths.quaternions import GetAttititudeJacobian, QuaternionToParam, GetLeftMatrix
from maths import linalg 



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

    def Linearize(self, ref_state: np.ndarray, ref_action: np.ndarray):
        context = self.quadrotor.CreateDefaultContext()
        context.SetContinuousState(ref_state)
        
        self.quadrotor.get_input_port().FixValue(context, ref_action)
        sys =   FirstOrderTaylorApproximation(self.quadrotor, 
                                context,
                                self.quadrotor.get_input_port().get_index(),
                                self.quadrotor.get_output_port().get_index())
        
        q0 = context.get_continuous_state_vector().CopyToVector()[:4].reshape((4,1))
        E = np.zeros((13, 12))
        E[:4, :3] = GetAttititudeJacobian(q0)
        E[4:, 3:] = np.eye(9)
        
        Ared = E.T @ sys.A() @ E
        Bred = E.T @ sys.B()

        return Ared, Bred
    

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

        quaternion_error = GetLeftMatrix(q_ref).T @ q.T

        if(np.any(quaternion_error[0]) <= 0.001):
            print(f"q_ref: {q_ref}\n q: {q}")

        differential_quaternion = QuaternionToParam(quaternion_error).T


        return np.hstack((differential_quaternion, state[:, 4:] - ref_state[4:]))

    
    def _OneStepRollout(self, x, u):
        context = self.plant.CreateDefaultContext()
        input_port = context.get_input_port()
        
        context.SetDiscreteState(x)
        input_port.FixValue(context, u)

        state = context.get_discrete_state()
        self.plant.CalcForcedDiscreteVariableUpdate(context, state)
        return state.get_vector().value().flatten()


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

        A, B = self.Linearize(ref_state, ref_action)
        self.K, _ = LinearQuadraticRegulator(A, B, self.Q, self.R)        

class QuadrotoriLQR(QuadrotorController):
    def __init__(self, quadrotor: Diagram, 
                 multibody_plant: MultibodyPlant, 
                 Q: np.ndarray, 
                 R: np.ndarray,
                 Qf: np.ndarray,
                 Tf: float,
                 dt: float):
        
        super().__init__(quadrotor, multibody_plant)
        self.Q: np.ndarray = Q
        self.R: np.ndarray = R
        self.Qf: np.ndarray = Qf
        self.Tf: float = Tf
        self.dt: float = dt

        self.nx = 12 #differential state size
        self.nu = 4
        self.num_time_steps: int = math.floor(Tf/dt) + 1

        self.xtraj: np.ndarray = np.zeros((self.num_time_steps, self.nx+1))
        self.utraj: np.ndarray = np.zeros((self.num_time_steps-1, self.nu))

    def InitTraj(self, x0: np.ndarray, utraj: np.ndarray):
        
        self.xtraj[:] = np.kron(np.ones(self.num_time_steps, 1), x0)
        self.utraj[:] = utraj

    def DoCalcVectorOutput(self, context: Context, motor_current: BasicVector):
        current_state = self.get_input_port(0).Eval(context)
        goal_state = self.get_input_port(1).Eval(context)

        if not self.xtraj.any() and not self.utraj.any():
            self.xtraj[:] = np.kron(np.ones((self.num_time_steps, 1)), current_state) #Maybe consider only passing x0 since this gets overwritten in self.control
            self.utraj[:] = 1000* np.random.randn(self.num_time_steps-1, self.nu)

        self.xtraj, self.utraj = self.control(current_state, goal_state, self.xtraj, self.utraj)
        motor_current.SetFromVector(self.utraj[0])

    def control(self, x0, xgoal, xtraj, utraj):
        self.xgoal = xgoal

        #Initial Rollout
        xtraj = self.Rollout(x0, utraj, self.dt)
        J = self.cost(xtraj, utraj)

        d = np.ones((self.num_time_steps-1, self.nu))            #feedforward

        iter = 0
        while np.max(np.linalg.norm(d, axis = 0)) > 1.5:
            iter += 1
            gradV = np.zeros((self.num_time_steps, self.nx))             #gradients
            hessV = np.zeros((self.num_time_steps, self.nx, self.nx))    #hessians 
            d[:] = np.ones((self.num_time_steps-1, self.nu))             #feedforward
            K = np.zeros((self.num_time_steps-1, self.nu, self.nx))      #feedback
            deltaJ = 0.0                                                 #change to trajectory cost             
            
            deltaJ, K, d[:] = self.backward_pass(xtraj, utraj, gradV, hessV, d, K)
            # Forward rollout with line search
            alpha = 1.0
            xtraj[:], utraj[:], Jn = self.forward_rollout(xtraj, utraj, d, K, alpha)
            while np.isnan(Jn) or Jn > (J - 1e-2 * alpha * deltaJ):
                alpha *= 0.5
                xtraj, utraj, Jn = self.forward_rollout(xtraj, utraj, d, K, alpha)
            J = Jn
            print(iter)
        return xtraj, utraj
    
    
    def backward_pass(self, xtraj, utraj, gradV, hessV, d, K):
        deltaJ = 0
        
        gradV[-1, :] = (self.Qf @ self._ComputeDifferentialState(np.expand_dims(xtraj[-1, :], 0),
                                                                            self.xgoal).T).squeeze()
        hessV[-1, :, :] = self.Qf

        for k in range(self.num_time_steps-1)[::-1]:
            xk, uk = xtraj[k, :], utraj[k, :] 

            q = self.Q @ self._ComputeDifferentialState(np.expand_dims(xk, 0), self.xgoal).T
            r = self.R @ uk

            # Linearization
            A, B = self.Linearize(xk, uk)

            # Action value gradient terms
            gx = q.squeeze() + A.T @ gradV[k+1, :]
            gu = r.squeeze() + B.T @ gradV[k+1, :]

            # Action value hessian terms
            Gxx = self.Q + A.T @ hessV[k+1, :, :] @ A
            Guu = self.R + B.T @ hessV[k+1, :, :] @ B
            Gxu = A.T @ hessV[k+1, :, :] @ B
            Gux = B.T @ hessV[k+1, :, :] @ A

            #Regularize
            beta = 0.1
            while not linalg.is_pos_def(self.full_hessian(Gxx, Gxu, Gux, Guu)):
                regularizer = beta * np.eye((A.shape[0]))

                Gxx += A.T @ regularizer @ A
                Guu += B.T @ regularizer @ B
                Gxu += A.T @ regularizer @ B
                Gux += B.T @ regularizer @ A
                
                beta *= 2

            # Feedforward term
            d[k, :] = linalg.qr_inverse(Guu) @ gu

            # Feedback term
            K[k, :, :] = linalg.qr_inverse(Guu) @ Gux

            #update action value gradient and hessian
            gradV[k, :] = (gx - K[k, :, :].T @ gu + K[k, :, :].T @ Guu @ d[k, :] - Gxu @ d[k, :])
            hessV[k, :, :] = Gxx + K[k, :, :].T @ Guu @ K[k, :, :] - K[k, :, :].T @ Gux
            
            deltaJ += gu.T @ d[k, :] 

        return deltaJ, K, d
    
    def forward_rollout(self, xtraj, utraj, d, K, alpha):
        xn = np.zeros((self.num_time_steps, self.nx+1))
        xn[0, :] = xtraj[0, :]
        un = np.zeros((self.num_time_steps-1, self.nu))

        for k in range(self.num_time_steps-1):
            un[k, :] = utraj[k, :] - alpha*d[k, :] - (K[k, :, :] @ self._ComputeDifferentialState(np.expand_dims(xn[k, :],0),
                                                                                                xtraj[k, :]).T).squeeze()
            # print(self._ComputeDifferentialState(np.expand_dims(xn[k, :],0),xtraj[k, :]).T)

            # print(f"Rollout {k} \n quat_norm {np.linalg.norm(xn[k, :4])} \n\n")
            # print(un[k, :])
            xn[k + 1, :] = self.Rollout(xn[k, :], np.expand_dims(un[k, :], 0), self.dt)[-1,:]

            # print(f"Rollout {k} \n State: {xn[k, :]} \n Action: {un[k, :]} \n\n")
        Jn = self.cost(xn, un)

        return xn, un, Jn

    def stage_cost(self, x: np.ndarray, u: np.ndarray):
        '''
        Computes cost due to state and action trajectory

        :param xtraj: state trajectory, ndarray of shape (num_time_steps-1, nx)
        :param utraj: action trajectory, ndarray of shape (num_time_steps-1, nu)
        '''
        xerr = self._ComputeDifferentialState(x, self.xgoal)
        # Weighted errors
        weighted_state_errors = xerr @ self.Q  # Shape (15, 12)
        weighted_action = u @ self.R

        state_cost = np.sum(weighted_state_errors * xerr) 
        action_cost = np.sum(weighted_action * u)
        return 0.5*state_cost + 0.5* action_cost

    def terminal_cost(self, xf: np.ndarray):
        '''
        Computes cost due to final state of state trajectory
        '''
        xerr = self._ComputeDifferentialState(xf, self.xgoal)
        
        return 0.5 * np.sum((xerr @ self.Qf) * xerr)


    def cost(self, xtraj: np.ndarray, utraj: np.ndarray):
        '''
        Computes total cost of trajectory
        :param xtraj: state trajectory, ndarray of shape (num_time_steps, nx)
        :param utraj: action trajectory, ndarray of shape (num_time_steps-1, nu)

        '''
        J = 0
        xf = np.expand_dims(xtraj[-1, :], 0)
        for k in range(self.num_time_steps-1):
            J += self.stage_cost(xtraj[:-1,:], utraj)

        J += self.terminal_cost(xf)
        return J

    def full_hessian(self, Gxx: np.ndarray, Gxu: np.ndarray, Gux: np.ndarray, Guu: np.ndarray)-> np.ndarray:
        '''
        Combines hessian parts into one
        '''
        G = np.vstack( 
            (np.hstack((Gxx, Gxu)),
             np.hstack((Gux, Guu)))
            )
        return G
    
if __name__ == "__main__":
    print("run")