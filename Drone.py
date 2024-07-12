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
)
from pydrake.examples import QuadrotorGeometry, QuadrotorPlant, StabilizingLQRController

import numpy as np
def quadrotor_example():
    builder = DiagramBuilder()

    plant = builder.AddSystem(QuadrotorPlant())

    controller = builder.AddSystem(StabilizingLQRController(plant, [0, 0, 1]))
    builder.Connect(controller.get_output_port(0), plant.get_input_port(0))
    builder.Connect(plant.get_output_port(0), controller.get_input_port(0))

    # Set up visualization in MeshCat
    scene_graph = builder.AddSystem(SceneGraph())
    QuadrotorGeometry.AddToBuilder(builder, plant.get_output_port(0), scene_graph)
    meshcat.Delete()
    meshcat.ResetRenderMode()
    meshcat.SetProperty("/Background", "visible", False)
    MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)
    # end setup for visualization

    diagram = builder.Build()

    # Set up a simulator to run this diagram
    simulator = Simulator(diagram)
    simulator.set_target_realtime_rate(0.0)
    context = simulator.get_mutable_context()

    # Simulate
    for i in range(5):
        context.SetTime(0.0)
        context.SetContinuousState(
            0.5
            * np.random.randn(
                12,
            )
        )
        simulator.Initialize()
        simulator.AdvanceTo(4.0)

def main():
    quadrotor_example()



if __name__ == '__main__':
    meshcat = StartMeshcat()

    main()

