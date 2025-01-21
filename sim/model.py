import numpy as np

class Model:
    def __init__(self):
        self.nx = None
        self.nu = None

    def get_dims(self):
        return self.nx, self.nu

    def discrete_dynamics(self, x, u):
        raise NotImplementedError

    def rollout(self, x0, u_trj):
        x_trj = np.zeros((u_trj.shape[0] + 1, x0.shape[0]))
        x_trj[0] = x0 
        for n in range(u_trj.shape[0]):
            x_trj[n+1] =self.discrete_dynamics(x_trj[n], u_trj[n])
        return x_trj      
    
    def _cost_stage(self, x, u):
       return NotImplementedError

    def _cost_final(self, x):
        return NotImplementedError

    def cost_trj(self, x_trj, u_trj):
        total = 0.0
        for i in range(x_trj.shape[0]-1):
            total += self._cost_stage (x_trj[i], u_trj[i])
        
        total += self._cost_final(x_trj[-1])
        return total    

    def stage(self, x, u):
        raise NotImplementedError
    
    def final(self, x):
        raise NotImplementedError    