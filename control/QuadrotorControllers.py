from pydrake.all import (VectorSystem,
                         Diagram,
                         MultibodyPlant,
                         FirstOrderTaylorApproximation,
                         LinearQuadraticRegulator,
                         Context)

import numpy as np

import os
import sys
sys.path.append(print(os.path.abspath(os.path.join(os.getcwd(), os.pardir))))

from math.quaternions import GetAttititudeJacobian, QuaternionToParam, GetLeftMatrix
from math.linalg import is_pos_def



class QuadrotorController(VectorSystem):
    """Base controller class for quadrotor using quaternion floating base
    """
    def __init__(self, quadrotor: Diagram, multibody_plant: MultibodyPlant):
            # 13 inputs (quadrotor state), 4 motor current outputs.
            VectorSystem.__init__(self, 13, 4)
            self.quadrotor = quadrotor
            self.plant = multibody_plant

    def DoCalcVectorOutput(self, context: Context, quadrotor_state, not_used, motor_current):
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


    def _ComputeDifferentialState(self, state: np.ndarray):
        '''
        Computes differential state accounting for quaternion kinematics
        '''
        q_ref = self.ref_state[:4]
        q = state[:4]
        differential_quaternion = QuaternionToParam(GetLeftMatrix(q_ref).T @ q.reshape((4,1))
                                                    ).reshape((3))
        return np.hstack((differential_quaternion, state[4:]- self.ref_state[4:]))
    



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

        mass = multibody_plant.CalcTotalMass(multibody_plant.GetMyContextFromRoot(quadrotor.CreateDefaultContext()))
        ref_state = np.array([1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.])
        gravity = multibody_plant.gravity_field().gravity_vector()[2]
        ref_action = np.array([-mass * gravity / 4 for i in range(4)] )
        self.SetReferencePoint(ref_state, ref_action)


    def SetReferencePoint(self, ref_state: np.ndarray, ref_action: np.ndarray):
        '''
        Updates reference state and action then recomputes linearization and optimal feedback gain
        :param ref_state: reference state consisting of 
        [qw, qx, qy, qz, x, y, z, wx, wy, wz, vx, vy, vz]^T
        
        :param ref_action: reference action consisting of 
        [motor_current_i for i in [0, 1, 2, 3]]
        '''
        self.ref_state = ref_state
        self.ref_action = ref_action

        A, B = self.Linearize(ref_state, ref_action)
        self.K, _ = LinearQuadraticRegulator(A, B, self.Q, self.R)


    def DoCalcVectorOutput(self, context: Context, quadrotor_state: np.ndarray, not_used: np.ndarray, motor_current: np.ndarray):
        differential_quadrotor_state = self._ComputeDifferentialState(quadrotor_state)
        motor_current[:] = self.ref_action - self.K @ differential_quadrotor_state   

class QuadrotoriLQR(QuadrotorController):
    def __init__(self, quadrotor: Diagram, 
                 multibody_plant: MultibodyPlant, 
                 Q: np.ndarray, 
                 R: np.ndarray,
                 Qf: np.ndarray,
                 Tf: float,
                 dt: float):
        
        super().__init__(quadrotor, multibody_plant)
        self.Q = Q
        self.R = R
        self.Qf = Qf
        self.Tf = Tf
        self.dt = dt

    def DoCalcVectorOutput(self, context: Context, quadrotor_state, not_used, motor_current):
        self.xtraj, self.utraj = self.control(quadrotor_state, self.xgoal, self.xtraj, self.utraj)
        motor_current[:] = self.utraj[0]

    def control(self, x0, xgoal, xtraj, utraj):
        self.xgoal = xgoal
        p = None        #gradients
        P= None         #hessians 
        d= None         #feedforward
        K= None         #feedback
        deltaJ= None    #change to trajectory cost 
        xn= None        #state traj
        un= None        #action traj
        gx= None        #gradient x term
        gu= None        #gradient u term
        Gxx= None       #Hessian xx term
        Guu= None       #Hessian uu term
        Gxu= None       #Hessian xu term
        Gux= None       #Hessian ux term


        #Initial rollout
        for k in range (1, self.N-1):
            xtraj[:, k+1] = self.dynamics_(xtraj[:, k], utraj[:, k])
        J = self.cost_(xtraj, utraj)

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

        p[:, self.N] = self.Qn_*(xtraj[:, -1] - self.xgoal)
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
        for k in range(1, self.N-1):
                un[:, k] = utraj[:, k] - alpha*d[k] - np.dot(K[:, :, k], xn[:, k] - xtraj[:, k])
                xn[:, k+1] = self.f_(xn[:, k], un[:, k])
        Jn = self.cost_(xn, un)

        return xn, un, Jn

    def stage_cost(self, x: np.ndarray, u: np.ndarray):
        '''
        Computes cost due to state and action trajectory
        '''
        return 0.5*((x - self.xgoal).T @ self.Q_ @ (x - self.xgoal)) + 0.5* u.T*self.R_*u

    def terminal_cost(self, xf: np.ndarray):
        '''
        Computes cost due to final state of state trajectory
        '''
        return xf.T@self.Qf_@xf


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
    pass