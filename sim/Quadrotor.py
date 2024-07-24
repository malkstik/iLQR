from IPython.display import SVG, display
import pydot

from pydrake.all import (
    Diagram, 
    MultibodyPlant,
    DiagramBuilder,
    Parser,
    PropellerInfo,
    RigidTransform,
    Propeller,
    LeafSystem,
    SceneGraph,
    AbstractValue,
    FramePoseVector,
    Context,
    Quaternion,
    OutputPort,
    )
import numpy as np

from pydrake.all import Variable
from sim.quaternions import hat, GetLeftMatrix, GetAttititudeJacobian, ParamToQuaternion, QuaternionToParam, QuaternionToRotation

class QuadrotorGeometry(LeafSystem):
    def __init__(self, scene_graph: SceneGraph) -> None:
        super().__init__()
        self.scene_graph = scene_graph
        
        #Add model
        plant = MultibodyPlant(0.0)
        parser = Parser(plant)
        (model_instance,) = parser.AddModelsFromUrl(
            "package://drake_models/skydio_2/quadrotor.urdf"
        )
        plant.RegisterAsSourceForSceneGraph(scene_graph)
        plant.Finalize()
        
        body_index = plant.GetBodyIndices(model_instance)[0]
        self.source_id = plant.get_source_id()
        self.frame_id_ = plant.GetBodyFrameIdOrThrow(body_index)
        self.DeclareVectorInputPort("state", 13)
        self.DeclareAbstractOutputPort(
            "geometry_pose",
            lambda: AbstractValue.Make(FramePoseVector()),
            self.OutputGeometryPose
        )
    
    def OutputGeometryPose(self, context: Context, 
                                poses: FramePoseVector):
        state = self.get_input_port(0).Eval(context)

        R = Quaternion(state[:4])
        p = state[4:7]
        pose = RigidTransform(R, p)
        FPV = FramePoseVector()
        FPV.set_value(self.frame_id_, pose)
        
        poses.set_value(FPV)
    
    @classmethod
    def AddToBuilder(cls, 
                    builder: DiagramBuilder,
                    quadrotor_state_port: OutputPort, 
                    scene_graph: SceneGraph) -> LeafSystem:
        quadrotor_geometry = builder.AddSystem(QuadrotorGeometry(scene_graph))
        quadrotor_geometry.set_name("QuadrotorGeometry")
        builder.Connect(
            quadrotor_state_port, quadrotor_geometry.get_input_port(0) 
        )
        builder.Connect(
            quadrotor_geometry.get_output_port(0),
            scene_graph.get_source_pose_port(quadrotor_geometry.source_id)
        )
        return quadrotor_geometry

# TODO: write linearization, potentially rewrite as vectorsystem
class AnalyticalQuadrotorModel:
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

        vars = ["x", "y", "z", "qw", "qx", "qy", "qz", "vx", "vy", "vz", "wx", "wy", "wz"]
        state = [Variable(var) for var in vars]
        x, y, z, qw, qx, qy, qz, vx, vy, vz, wx, wy, wz = state
        dynamics = [
            
        ]
        

    def continuous_dynamics(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        '''
        :param x: state as [x, y, z, qw, qx, qy, qz, vx, vy, vz, wx, wy, wx]^T
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




def MakeMultibodyQuadrotor(show_diagram = False) -> tuple[Diagram, MultibodyPlant]:
    builder = DiagramBuilder()
    # The MultibodyPlant handles f=ma, but doesn't know about propellers.
    plant: MultibodyPlant = builder.AddSystem(MultibodyPlant(0.0))
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
    builder.ExportOutput(plant.get_state_output_port(), "y")

    diagram: Diagram = builder.Build()

    if show_diagram:
        display(SVG(pydot.graph_from_dot_data(
            diagram.GetGraphvizString(max_depth=2))[0].create_svg()))
    return diagram, plant


if __name__ == "__main__":
    quadrotor, mbp = MakeMultibodyQuadrotor()

