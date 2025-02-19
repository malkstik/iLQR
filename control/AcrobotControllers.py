from control.models.acrobot import Acrobot2DModel
from pydrake.all import LeafSystem, DiscreteTimeLinearQuadraticRegulator, LinearQuadraticRegulator
from control.iLQRController import iLQR
import numpy as np

class AcrobotiLQRController(LeafSystem):
    def __init__(self, acrobot,
                        Q,
                        Qf,
                        R,
                        dt,
                        N,         
                        max_iter,
                        regu_init,
                        min_regu,
                        max_regu,
                        max_linesearch_iters,
                        **kwargs
                ):
        LeafSystem.__init__(self)
        self.acrobot = acrobot
        model = Acrobot2DModel( Q,
                                Qf,
                                R,
                                dt,
                                N)

        self.controller = iLQR( model,
                                N,         
                                max_iter,
                                regu_init,
                                min_regu,
                                max_regu,
                                max_linesearch_iters,
                                )


        self.N = N
        self.nx, self.nu = model.get_dims()

        self.xtraj: np.ndarray = np.zeros((self.N, self.nx))
        self.utraj: np.ndarray = np.zeros((self.N-1, self.nu))
        self.uref: np.ndarray = np.zeros(self.nu, dtype = np.float64)


        self.DeclareVectorInputPort("current_state", self.nx)
        self.DeclareVectorInputPort("goal_state", self.nx)
        self.DeclareVectorInputPort("ref_action", self.nu)
        self.DeclareVectorOutputPort("control_output", self.nu, self.DoCalcVectorOutput) 

    
    def DoCalcVectorOutput(self, context, motor_current):
        current_state = self.get_input_port(0).Eval(context)
        goal_state = self.get_input_port(1).Eval(context)

        if self.controller.model.xref is not goal_state:
            self.controller.model.set_references(goal_state, self.uref)

        if not self.utraj.any():
            print("Init utraj")
            #Normally distribute about a hover @ identity rotation
            self.utraj[:] = np.random.randn(self.N-1, self.nu)*2.0
        else:
            #Use previous utraj as initial guess with shifting forward a single timestep, 
            self.utraj[:-1, :] = self.utraj[1:,:]

        self.xtraj, self.utraj = self.controller.run_ilqr(current_state, self.utraj)

        motor_current.SetFromVector(self.utraj[0, :])
    
    def CreateDefaultContext(self):
        return self.acrobot.CreateDefaultContext()    
    
class AcrobotLQRController(LeafSystem):
    def __init__(self,  Q,
                        Qf,
                        R,
                        dt,
                        N,
                        mode = "discrete",
                        **kwargs         
                ):
        LeafSystem.__init__(self)
        self.model = Acrobot2DModel( Q,
                                Qf,
                                R,
                                dt,
                                N)
        self.N = N
        self.nx, self.nu = self.model.get_dims()

        self.xtraj: np.ndarray = np.zeros((self.N, self.nx))
        self.utraj: np.ndarray = np.zeros((self.N-1, self.nu))
        self.uref: np.ndarray = np.zeros(self.nu, dtype = np.float64)

        self.mode = mode

        self.DeclareVectorInputPort("current_state", self.nx)
        self.DeclareVectorInputPort("goal_state", self.nx)
        self.DeclareVectorInputPort("ref_action", self.nu)
        self.DeclareVectorOutputPort("control_output", self.nu, self.DoCalcVectorOutput) 

    
    def DoCalcVectorOutput(self, context, motor_current):
        current_state = self.get_input_port(0).Eval(context)
        goal_state = self.get_input_port(1).Eval(context)

        if self.mode == "discrete":
            A, B = self.model._linearize_discrete(goal_state, 0)
            K, _ = DiscreteTimeLinearQuadraticRegulator(A, B, self.model.Q, self.model.R)

        else:
            A, B = self.model._linearize_continuous(goal_state, 0)
            K, _ = LinearQuadraticRegulator(A, B, self.model.Q, self.model.R)

        u = - K @ (current_state - goal_state)

        motor_current.SetFromVector(u)
    