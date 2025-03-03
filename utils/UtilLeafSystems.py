from pydrake.all import LeafSystem, BasicVector
import numpy as np

class NonConstantVectorSource(LeafSystem):
    def __init__(self, state_dim):
        LeafSystem.__init__(self)
        self.DeclareVectorOutputPort("output", BasicVector(state_dim), self.DoCalcOutput)
        self.state = np.zeros(state_dim)

    def SetState(self, new_state):
        self.state = new_state

    def DoCalcOutput(self, context, output):
        output.SetFromVector(self.state)