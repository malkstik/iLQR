from pydrake.all import (VectorSystem,
                         Diagram,
                         MultibodyPlant,
                         FirstOrderTaylorApproximation,
                         LinearQuadraticRegulator)

import numpy as np

import os
import sys
sys.path.append(print(os.path.abspath(os.path.join(os.getcwd(), os.pardir))))


from sim.quaternions import GetAttititudeJacobian, QuaternionToParam, GetLeftMatrix



class QuadrotorLQR(VectorSystem):
    """Define LQR controller for quadrotor using quaternion floating base
    """

    def __init__(self, quadrotor: Diagram, multibody_plant: MultibodyPlant, Q: np.ndarray, R: np.ndarray):
        # 13 inputs (quadrotor state), 4 motor current outputs.
        VectorSystem.__init__(self, 13, 4)
        self.quadrotor = quadrotor
        self.plant = multibody_plant
        self.Q = Q
        self.R = R

        mass = multibody_plant.CalcTotalMass(multibody_plant.GetMyContextFromRoot(quadrotor.CreateDefaultContext()))
        ref_state = np.array([1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.])
        gravity = multibody_plant.gravity_field().gravity_vector()[2]
        ref_action = np.array([-mass * gravity / 4 for i in range(4)] )
        self.SetReferencePoint(ref_state, ref_action)


    def SetReferencePoint(self, ref_state, ref_action):
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

    def DoCalcVectorOutput(self, context, quadrotor_state, not_used, motor_current):
        differential_quadrotor_state = self._ComputeDifferentialState(quadrotor_state)
        motor_current[:] = self.ref_action - self.K @ differential_quadrotor_state   


    def _ComputeFeedbackGain(self, context):
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

    def _ComputeDifferentialState(self, state):
        q_ref = self.ref_state[:4]
        q = state[:4]
        differential_quaternion = QuaternionToParam(GetLeftMatrix(q_ref).T @ q.reshape((4,1))
                                                    ).reshape((3))
        return np.hstack((differential_quaternion, state[4:]- self.ref_state[4:]))


if __name__ == "__main__":
    pass