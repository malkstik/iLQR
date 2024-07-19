import numpy as np
from pydrake.all import (
    DiagramBuilder,
    LinearQuadraticRegulator,
    MeshcatVisualizer,
    MultibodyPlant,
    Parser,
    Propeller,
    PropellerInfo,
    RigidTransform,
    RobotDiagramBuilder,
    SceneGraph,
    Simulator,
    Meshcat,
    StartMeshcat,
    namedview,
    FirstOrderTaylorApproximation)

from IPython.display import SVG, display
import pydot

from quaternions import hat, GetLeftMatrix, GetAttititudeJacobian, ParamToQuaternion, QuaternionToParam, QuaternionToRotation


class Quadrotor:
    '''
    Quadrotor built using Multibody plant, based on underactuated example:
    https://deepnote.com/workspace/Underactuated-2ed1518a-973b-4145-bd62-1768b49956a8/project/096cffe7-416e-4d51-a471-5fd526ec8fab/notebook/quadrotor-ac8fa093ca5a4894a618ada316f2750b
    
    Key difference is the continued use of quaternions instead of RPY
    
    :param L:   length of quadrotor arms (distance from propellor to COM) in meters
    :param kF:  force constant
    :param kM:  torque constant
    '''
    def __init__(self, L: float = 0.15, kF: float = 1.0, kM: float = 0.0245):
        self.builder = DiagramBuilder()
        self.plant = self.builder.AddSystem(MultibodyPlant(0.0))

        parser = Parser(self.plant)
        (model_instance,) = parser.AddModelsFromUrl(
            "package://drake_models/skydio_2/quadrotor.urdf"
        )
        self.plant.Finalize()

        body_index = self.plant.GetBodyByName("base_link").index()

        # Default parameters from quadrotor_plant.cc

        prop_info = [
            PropellerInfo(body_index, RigidTransform([L, 0, 0]), kF, kM),
            PropellerInfo(body_index, RigidTransform([0, L, 0]), kF, -kM),
            PropellerInfo(body_index, RigidTransform([-L, 0, 0]), kF, kM),
            PropellerInfo(body_index, RigidTransform([0, -L, 0]), kF, -kM),
        ]
        propellers = self.builder.AddSystem(Propeller(prop_info))
        self.builder.Connect(
            propellers.get_output_port(),
            self.plant.get_applied_spatial_force_input_port(),
        )
        self.builder.Connect(
            self.plant.get_body_poses_output_port(),
            propellers.get_body_poses_input_port(),
        )
        self.builder.ExportInput(propellers.get_command_input_port(), "u")

        self.diagram = self.builder.Build()

    def get_diagram(self):
        '''
        returns diagram as an SVG, which can be saved or viewed using iPython.display() in a notebook
        '''
        return SVG(pydot.graph_from_dot_data(
            self.diagram.GetGraphvizString(max_depth=2))[0].create_svg())

class QuadrotorModel:
    def __init__(self, L, kF, kM, m, J):
        self.L = L
        self.kF = kF
        self.kM = kM
        self.m = m
        self.J = J


        self.J_inv = np.linalg.inv(J)
        self.kF_matrix = np.vstack((np.zeros((2,4)), kF * np.zeros((1, 4))))
        self.kM_matrix = np.array([[   0,   L*kF,   0  , -L*kF],
                                   [-L*kF,    0,   L*kF,   0],
                                   [  kM,   -kM,    kM,   -kM]])
        self._gravity = np.array([0, 0, -9.80665]).reshape((3,1))


        

    def continuous_dynamics(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        '''
        :param x: state as [x, y, z, qw, qx, qy, qz, vx, vy, vx, wx, wy, wx]^T
        :param u: action as motor currents
        :return x_dot: state time derivative
        '''

        #State
        r = x[:3]
        q = x[3:7]
        v = x[7:10]
        w = x[10:]

        Q = QuaternionToRotation(q)

        # Time derivatives
        r_dot = Q @ r
        q_dot = 0.5 * GetAttititudeJacobian(q) @ w
        v_dot = Q.T @ self._gravity + 1/self.m * self.kF_matrix @ u - hat(w) @ v
        w_dot = self.J_inv @ (-hat(w) @ self.J @ w + self.kM_matrix @ u) 

        return np.hstack((r_dot, q_dot, v_dot, w_dot))

    def augment_matrix(self, q):
        E = np.zeros((13, 12))
        E[:3, :3] = np.eye(3)
        E[3:7, 3:6] = GetAttititudeJacobian(q)
        E[7:, 6:] = np.eye(6)

        return E 

    def linearize_dynamics(self, reference_state, reference_action):
        q0 = reference_state[3:7]
        #Linearize about reference
        A, B = None, None
        #Compute reduced system
        E = self.augment_matrix(q0)
        A_reduced = E.T @ A @ E
        B_reduced = E.T @ B

        return A_reduced, B_reduced

def MakeMultibodyQuadrotor():
    builder = DiagramBuilder()
    # The MultibodyPlant handles f=ma, but doesn't know about propellers.
    plant = builder.AddSystem(MultibodyPlant(0.0))
    parser = Parser(plant)
    (model_instance,) = parser.AddModelsFromUrl(
        "package://drake_models/skydio_2/quadrotor.urdf"
    )
    plant.Finalize()

    # Now we can add in propellers as an external force on the MultibodyPlant.
    body_index = plant.GetBodyByName("base_link").index()
    # Default parameters from quadrotor_plant.cc:
    L = 0.15  # Length of the arms (m).
    kF = 1.0  # Force input constant.
    kM = 0.0245  # Moment input constant.

    # Note: Rotors 0 and 2 rotate one way and rotors 1 and 3 rotate the other.
    prop_info = [
        PropellerInfo(body_index, RigidTransform([L, 0, 0]), kF, kM),
        PropellerInfo(body_index, RigidTransform([0, L, 0]), kF, -kM),
        PropellerInfo(body_index, RigidTransform([-L, 0, 0]), kF, kM),
        PropellerInfo(body_index, RigidTransform([0, -L, 0]), kF, -kM),
    ]
    propellers = builder.AddSystem(Propeller(prop_info))
    builder.Connect(
        propellers.get_output_port(),
        plant.get_applied_spatial_force_input_port(),
    )
    builder.Connect(
        plant.get_body_poses_output_port(),
        propellers.get_body_poses_input_port(),
    )
    builder.ExportInput(propellers.get_command_input_port(), "u")

    return builder.Build(), plant