
from control.models.Quadrotor_EA import QuadrotorEAModel
from control.models.Quadrotor_Quat import QuadrotorQuatModel

from pydrake.all import (LeafSystem,
                         Diagram,
                         MultibodyPlant,
                         Context,
                         BasicVector,
                         DiscreteTimeLinearQuadraticRegulator,
                         LinearQuadraticRegulator)

import numpy as np 
from control.iLQRController import iLQR


class QuadrotorController_EA(LeafSystem):
    """controller base class for quadrotor using euler (RPY) angle floating base
    """
    def __init__(self,  Q,
                        Qf,
                        R,
                        dt,
                        N,         
                        max_iter,
                        max_linesearch_iters,
                        d_tol,
                    ):
        # 12 inputs (quadrotor state), 4 motor current outputs.
        LeafSystem.__init__(self)
        self.model = QuadrotorEAModel(  Q,
                                        Qf,
                                        R,
                                        dt,
                                        N)

        self.controller = iLQR(self.model,
                                N,         
                                max_iter,
                                max_linesearch_iters,
                                d_tol
                                )

        self.N = N
        self.nx, self.nu = self.model.get_dims()
        self.dt = dt
        self.last_solve = 0

        self.xtraj: np.ndarray = np.zeros((self.N, self.nx))
        self.utraj: np.ndarray = np.zeros((self.N-1, self.nu))
        self.uref: np.ndarray = np.zeros(4, dtype = np.float64)


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

        if context.get_time() == 0 or context.get_time()-self.last_solve >= self.dt:
            #Use previous utraj as initial guess with shifting forward a single timestep, 
            self.utraj[:-1, :] = self.utraj[1:,:]
            self.xtraj, self.utraj = self.controller.run_ilqr(current_state, self.utraj)
            self.last_solve = context.get_time()

        motor_current.SetFromVector(self.utraj[0, :])
    
    def CreateDefaultContext(self):
        return self.quadrotor.CreateDefaultContext()    
    

class QuadrotorEALQRController(LeafSystem):
    def __init__(self,  Q,
                        Qf,
                        R,
                        dt,
                        N,
                        mode = "discrete",
                        **kwargs         
                ):
        LeafSystem.__init__(self)
        self.model = QuadrotorEAModel( Q,
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
        

class QuadrotorController_Quat(LeafSystem):
    """controller base class for quadrotor using euler (RPY) angle floating base
    """
    def __init__(self,  Q,
                        Qf,
                        R,
                        dt,
                        N,         
                        max_iter,
                        max_linesearch_iters,
                        d_tol,
                    ):
        # 12 inputs (quadrotor state), 4 motor current outputs.
        LeafSystem.__init__(self)
        self.model = QuadrotorEAModel(  Q,
                                        Qf,
                                        R,
                                        dt,
                                        N)

        self.controller = iLQR(self.model,
                                N,         
                                max_iter,
                                max_linesearch_iters,
                                d_tol
                                )

        self.N = N
        self.nx, self.nu = self.model.get_dims()
        self.dt = dt
        self.last_solve = 0

        self.xtraj: np.ndarray = np.zeros((self.N, self.nx))
        self.utraj: np.ndarray = np.zeros((self.N-1, self.nu))
        self.uref: np.ndarray = np.zeros(4, dtype = np.float64)


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

        if context.get_time() == 0 or context.get_time()-self.last_solve >= self.dt:
            #Use previous utraj as initial guess with shifting forward a single timestep, 
            self.utraj[:-1, :] = self.utraj[1:,:]
            self.xtraj, self.utraj = self.controller.run_ilqr(current_state, self.utraj)
            self.last_solve = context.get_time()

        motor_current.SetFromVector(self.utraj[0, :])
    
    def CreateDefaultContext(self):
        return self.quadrotor.CreateDefaultContext()    
    

class QuadrotorQuatLQRController(LeafSystem):
    def __init__(self,  Q,
                        R,
                        mode = "discrete",
                        **kwargs         
                ):
        LeafSystem.__init__(self)
        self.model = QuadrotorQuatModel(Q,
                                Q,
                                R,
        )
        self.nx, self.nu = self.model.get_dims()

        self.uref: np.ndarray = np.array([1.9006875, 1.9006875, 1.9006875, 1.9006875])

        self.mode = mode

        self.DeclareVectorInputPort("current_state", self.nx)
        self.DeclareVectorInputPort("goal_state", self.nx)
        self.DeclareVectorInputPort("ref_action", self.nu)
        self.DeclareVectorOutputPort("control_output", self.nu, self.DoCalcVectorOutput) 

    def DoCalcVectorOutput(self, context, motor_current):
        current_state = self.get_input_port(0).Eval(context)
        goal_state = self.get_input_port(1).Eval(context)

        if self.mode == "discrete":
            A, B = self.model._linearize_discrete(goal_state, self.uref)
            K, _ = DiscreteTimeLinearQuadraticRegulator(A, B, self.model.Q, self.model.R)

        else:
            A, B = self.model._linearize_continuous(goal_state, self.uref)
            K, _ = LinearQuadraticRegulator(A, B, self.model.Q, self.model.R)

        dx = self.model.CalcDifferentialState(current_state, goal_state)

        motor_current.SetFromVector((self.uref - K @ dx).squeeze())
if __name__ == "__main__":
    print("run")