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

