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
from maths.linalg import is_pos_def



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

    def Linearize(self, ref_state, ref_action):
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
    

    def Rollout(self, x0: np.ndarray, U: np.ndarray, dt: float):
        """
        Simulates the trajectory of states (rollout) for given initial state and control sequence.
        """
        X = [x0.copy().tolist()]

        simulator = Simulator(self.quadrotor)
        simulator_context = simulator.get_mutable_context()
        simulator_context.SetContinuousState(x0)

        for u in U:
            # Set input and simulate a small time step

            self.quadrotor.get_input_port().FixValue(simulator_context, u)
            # self.quadrotor.get_continuous_state_output_port().Calc(context, context.get_mutable_continuous_state_vector())
            
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
    
    def _ComputeDifferentialState(self, state: np.ndarray, ref_state: np.ndarray):
        '''
        Computes differential state accounting for quaternion kinematics
        '''
        q_ref = ref_state[:4]
        q = state[:4]
        differential_quaternion = QuaternionToParam(GetLeftMatrix(q_ref).T @ q.reshape((4,1))
                                                    ).reshape((3))
        return np.hstack((differential_quaternion, state[4:]- ref_state[4:]))
    

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

        differential_quadrotor_state = self._ComputeDifferentialState(current_state, self.ref_state)
        # motor_current[:] = self.ref_action - self.K @ differential_quadrotor_state   
        motor_current.SetFromVector(self.ref_action - self.K @ differential_quadrotor_state)
        
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
            self.xtraj[:] = np.kron(np.ones((self.num_time_steps, 1)), current_state)
            self.utraj[:] = np.random.randn(self.num_time_steps-1, self.nu)
        self.xtraj, self.utraj = self.control(current_state, goal_state, self.xtraj, self.utraj)
        motor_current.SetFromVector(self.utraj[0])

    def control(self, x0, xgoal, xtraj, utraj):
        self.xgoal = xgoal

        p = np.zeros((self.nx, self.num_time_steps))            #gradients
        P= np.zeros((self.nx, self.nx, self.num_time_steps))    #hessians 
        d= np.ones((self.num_time_steps-1))                     #feedforward
        K= np.zeros((self.nu, self.nx, self.num_time_steps-1))  #feedback
        deltaJ= 0.0                                             #change to trajectory cost 
        xn= np.zeros((self.nx, self.num_time_steps))            #state traj
        un= np.zeros((self.nu, self.num_time_steps))            #action traj
        # gx= np.zeros((self.nx))                                 #gradient x term
        # gu= np.zeros((self.nu))                                 #gradient u term
        # Gxx= np.zeros((self.nx, self.nx))                       #Hessian xx term
        # Guu= np.zeros((self.nu, self.nu))                       #Hessian uu term
        # Gxu= np.zeros((self.nx, self.nu))                       #Hessian xu term
        # Gux= np.zeros((self.nu, self.nx))                       #Hessian ux term


        #Initial Rollout
        # context = self.quadrotor.CreateDefaultContext()
        xtraj = self.Rollout(x0, utraj, self.dt)
        J = self.cost(xtraj, utraj)


        while np.maximum(np.abs(d)) > 1e-3:

            deltaJ, K = self.backward_pass_(xn, un, p, P, d, K)

            # Forward rollout with line search
            xn[:, 1] = xtraj[:, 1]
            alpha = 1.0

            xn, un, Jn = self.forward_rollout(self, xtraj, utraj, d, K, alpha)
            while np.isnan(Jn) or Jn > (J - 1e-2 * alpha * deltaJ):
                xn, un, Jn = self.forward_rollout(self, xtraj, utraj, d, K, alpha)

            J = Jn
            xtraj = xn
            utraj = un
        
        return xtraj, utraj
    
    
    def backward_pass(self, xtraj, utraj, p, P, d, K):
        deltaJ = 0

        p[:, self.N] = self.Qn_ @ (self._ComputeDifferentialState(xtraj[:, -1], self.xgoal)).T
        P[:, :, self.N] = self.Qn_

        for k in range(self.N)[::-1]:
            xk, uk = xtraj[:, k], utraj[:, k] 
            q = self.Q_ @ (xk-self.xgoal)
            r = self.R_ @ uk

            # Linearization
            A, B = self.Linearize(xk, uk)

            # Action value gradient terms
            gx = q + A.T @ p[:, k+1]
            gu = r + B.T @ p[:, k+1]

            # Action value hessian terms
            Gxx = self.Q + A.T @ P[:, :, k+1] @ A
            Guu = self.R + B.T @ P[:, :, k+1] @ B
            Gxu = A.T @ P[:, :, k+1] @ B
            Gux = B.T @ P[:, :, k+1] @ A

            #Regularize
            beta = 0.1
            while not is_pos_def(self.full_hessian_(Gxx, Gxu, Gux, Guu)):
                Gxx += beta * A.T @ A
                Guu += beta * B.T @ B
                Gxu += beta * A.T @ B
                Gux += beta * B.T @ A
                beta *= 2
                
            # Feedforward term
            d[k] = np.linalg.inv(Guu) @ gu
            # Feedback term
            K[:, :, k] = np.linalg.inv(Guu) @ Gux

            #update p and P
            p[:, k] = (gx - K[:, :, k].T @ gu + K[:, :, k].T @ Guu @ d[k] - Gxu @ d[k]).squeeze()
            P[:, :, k] = Gxx + K[:, :, k].T @ Guu @ K[:, :, k] - K[:, :, k].T @ Gux
            
            deltaJ += gu.T @ d[k] 
        return deltaJ, K
    
    def forward_rollout(self, xtraj, utraj, d, K, alpha):
        xn = np.zeros(self.nx_, self.N)
        un = np.zeros(self.nu_, self.N-1)
        context = self.CreateDefaultContext()
        for k in range(1, self.N-1):
                un[:, k] = utraj[:, k] - alpha*d[k] - np.dot(K[:, :, k], self._ComputeDifferentialState(xn[:, k], xtraj[:, k]))
                xn[:, k+1] = self.Rollout(context, xn[:, k], un[:,k], self.dt)
        Jn = self.cost_(xn, un)

        return xn, un, Jn

    def stage_cost(self, x: np.ndarray, u: np.ndarray):
        '''
        Computes cost due to state and action trajectory
        '''
        
        xerr = self._ComputeDifferentialState(x, self.xgoal)
        return 0.5*(xerr.T @ self.Q_ @ xerr) + 0.5* u.T*self.R_*u

    def terminal_cost(self, xf: np.ndarray):
        '''
        Computes cost due to final state of state trajectory
        '''
        xerr = self._ComputeDifferentialState(x, self.xgoal)
        return xerr.T @ self.Qf @ xerr


    def cost(self, xtraj: np.ndarray, utraj: np.ndarray):
        '''
        Computes total cost of trajectory
        '''
        J = 0
        xf = xtraj[:, -1]
        for k in range(self.N-1):
            J += self.stage_cost_(xtraj[:,:-1], utraj)
        J += self.terminal_cost_(xf)

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