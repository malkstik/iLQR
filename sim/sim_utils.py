from pydrake.all import (
    BallRpyJoint,
    Box,
    CoulombFriction,
    Cylinder,
    PrismaticJoint,
    RevoluteJoint,
    RigidTransform,
    SpatialInertia,
    Sphere,
    UnitInertia,
)


def RK4(dynamics, x, u, h = 0.001):
    '''
    Fourth order Runge-Kutta explicit integration

    :param dynamics: function that returns x_dot as a function of state and input
    :param x: state
    :oaram u: action
    :return : state at time t + h
    '''
    f1 = dynamics(x, u)
    f2 = dynamics(x + 0.5*h*f1, u)
    f3 = dynamics(x + 0.5*h*f2, u)
    f4 = dynamics(x + 1.0*h*f3, u)
    return x + (h/ 6.0)*(f1 + 2*f2 + 2*f3 + f4)

#From inside underactuated container:
# https://hub.docker.com/layers/russtedrake/underactuated/latest/images/sha256-f2f62490d1a18ee4ed8469630a43caf0c33193f29ba689003717a306b3734a5a?context=explore
def AddFloatingRpyJoint(plant, frame, instance, use_ball_rpy=True):
    inertia = SpatialInertia(
        mass=0, p_PScm_E=[0.0, 0.0, 0.0], G_SP_E=UnitInertia(0, 0, 0)
    )
    x_body = plant.AddRigidBody("x", instance, inertia)
    plant.AddJoint(
        PrismaticJoint(
            "x", plant.world_frame(), x_body.body_frame(), [1, 0, 0]
        )
    )
    y_body = plant.AddRigidBody("y", instance, inertia)
    plant.AddJoint(
        PrismaticJoint(
            "y", x_body.body_frame(), y_body.body_frame(), [0, 1, 0]
        )
    )
    z_body = plant.AddRigidBody("z", instance, inertia)
    plant.AddJoint(
        PrismaticJoint(
            "z", y_body.body_frame(), z_body.body_frame(), [0, 0, 1]
        )
    )
    if use_ball_rpy:
        plant.AddJoint(BallRpyJoint("ball", z_body.body_frame(), frame))
    else:
        # RollPitchYaw is body z-y-x
        rz_body = plant.AddRigidBody("rz", instance, inertia)
        plant.AddJoint(
            RevoluteJoint(
                "rz", z_body.body_frame(), rz_body.body_frame(), [0, 0, 1]
            )
        )
        ry_body = plant.AddRigidBody("ry", instance, inertia)
        plant.AddJoint(
            RevoluteJoint(
                "ry", rz_body.body_frame(), ry_body.body_frame(), [0, 1, 0]
            )
        )
        plant.AddJoint(
            RevoluteJoint("rx", ry_body.body_frame(), frame, [1, 0, 0])
        )