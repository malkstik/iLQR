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

from sim.quaternions import GetAttititudeJacobian, QuaternionToParam, GetLeftMatrix

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
        context = self.quadrotor.CreateDefaultContext()
        context.SetContinuousState(ref_state)
        
        self.quadrotor.get_input_port().FixValue(context, ref_action)

        self.ref_state = ref_state
        self.ref_action = ref_action

        self._ComputeFeedbackGain(context)

    def DoCalcVectorOutput(self, context: Context, quadrotor_state: np.ndarray, not_used: np.ndarray, motor_current: np.ndarray):
        differential_quadrotor_state = self._ComputeDifferentialState(quadrotor_state)
        motor_current[:] = self.ref_action - self.K @ differential_quadrotor_state   

    def _ComputeFeedbackGain(self, context: Context):
        '''
        Computes optimal feedback gain matrix from LQR
        '''
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

        self.K, _ = LinearQuadraticRegulator(Ared, Bred, self.Q, self.R)

if __name__ == "__main__":
    pass