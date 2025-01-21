
from sim.Quadrotor_EA import QuadrotorEAModel


from pydrake.all import (LeafSystem,
                         Diagram,
                         MultibodyPlant,
                         Context,
                         BasicVector)

import numpy as np 
from control.iLQRController import iLQR


class QuadrotorController_EA(LeafSystem):
    """controller base class for quadrotor using euler (RPY) angle floating base
    """
    def __init__(self, quadrotor: Diagram, 
                       multibody_plant: MultibodyPlant,
                        Q,
                        Qf,
                        R,
                        dt,
                        N,         
                        max_iter,
                        regu_init,
                        min_regu,
                        max_regu,
                        max_linesearch_iters                   
                    ):
        # 12 inputs (quadrotor state), 4 motor current outputs.
        LeafSystem.__init__(self)
        self.quadrotor = quadrotor
        self.plant = multibody_plant

        self.mass = multibody_plant.CalcTotalMass(multibody_plant.GetMyContextFromRoot(quadrotor.CreateDefaultContext()))

        self.model = QuadrotorEAModel(quadrotor, 
                                        multibody_plant,
                                        Q,
                                        Qf,
                                        R,
                                        dt)
        self.controller = iLQR(self.model,
                                N,         
                                max_iter,
                                regu_init,
                                min_regu,
                                max_regu,
                                max_linesearch_iters)

        self.N = N
        self.nx, self.nu = self.model.get_dims()

        self.xtraj: np.ndarray = np.zeros((self.N, self.nx))
        self.utraj: np.ndarray = np.zeros((self.N-1, self.nu))
        self.uref: np.ndarray = np.zeros(4, dtype = np.float64)


        self.DeclareVectorInputPort("current_state", self.nx)
        self.DeclareVectorInputPort("goal_state", self.nx)
        self.DeclareVectorInputPort("ref_action", self.nu)
        self.DeclareVectorOutputPort("control_output", self.nu, self.DoCalcVectorOutput) 

    
    def DoCalcVectorOutput(self, context: Context, motor_current: BasicVector):
        current_state = self.get_input_port(0).Eval(context)
        goal_state = self.get_input_port(1).Eval(context)

        if self.model.xref is not goal_state:
            self.model.set_references(goal_state, self.uref)

        if not self.xtraj.any() and not self.utraj.any():
            self.utraj[:] = 1.95* np.ones((self.N-1, self.nu)) #Normally distribute about a hover @ identity rotation

        self.xtraj, self.utraj = self.controller.run_ilqr(current_state, self.utraj)
        motor_current.SetFromVector(self.utraj[0])
    
    def CreateDefaultContext(self):
        return self.quadrotor.CreateDefaultContext()    