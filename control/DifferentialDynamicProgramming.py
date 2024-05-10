from ..sim.sim_utils import RK4

import numpy as np

import jax
import jax.numpy as jnp


def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)

class iLQR:
    def __init__(self, dynamics, nx, nu, Q, R, Qf, Tf = 5.0, h = 0.001):
        self.dynamics_ = dynamics
        self.nx_ = nx
        self.nu_ = nu        

        self.Tf_ = Tf
        self.h_ = h
        self.N_ = int(self.Tf_/self.h_) + 1
        self.thist = np.arange(0, self.N_-1, self.h_)

        self.Q_ = Q
        self.R_ = R
        self.Qf_ = Qf

        self.f_ = lambda x, u: RK4(dynamics, x, u, h = self.h_)

        self.dfdx_ = jax.grad(self.f_, argnums=0)
        self.dfdu_ = jax.grad(self.f_, argnums=1)

        self.dAdx_ = None
        self.dBdx_ = None
        self.dAdu_ = None
        self.dBdu_ = None

        # self.dAdx_ = jax.grad(self.dynamics_, )
        # self.dBdx_ = jax.grad(self.dynamics_, )
        # self.dAdu_ = jax.grad(self.dynamics_, )
        # self.dBdu_ = jax.grad(self.dynamics_, )

        self.xgoal = None

    def stage_cost_(self, x: np.ndarray, u: np.ndarray):
        return 0.5*((x - self.xgoal).T @ self.Q_ @ (x - self.xgoal)) + 0.5* u.T*self.R_*u

    def terminal_cost_(self, xf):
        return xf.T@self.Qf_@xf

    def cost_(self, xtraj, utraj):
        J = 0
        xf = xtraj[:, -1]
        for k in range(self.N_-1):
            J += self.stage_cost_(xtraj[:,:-1], utraj)
        J += self.terminal_cost_(xf)

        return J
    

    def regularize(self, Gxx, Gxu, Gux, Guu):
        pass

    def backward_pass_(self, xtraj, utraj, p, P, d, K):
        gradJ = 0

        p[:, self.N_] = self.Qn_*(xtraj[:, -1] - self.xgoal)
        P[:, :, self.N_] = self.Qn_

        for k in range(self.N_)[::-1]:
            xk, uk = xtraj[:, k], utraj[:, k] 
            q = self.Q_ @ (xk-self.xgoal)
            r = self.R_ @ uk

            A = self.dfdx_(xk, uk)
            B = self.dfxu_(xk, uk)

            gx = q + A.T@p[:, k+1]
            gu = r + B.T@p[:, k+1]

            Gxx = self.Q_ + A.T@P[:, :, k+1]@A
            Guu = self.R_ + B.T@P[:, :, k+1]@B
            Gxu = A.T @ P[:, :, k+1] @ B
            Gux = B.T @ P[:, :, k+1] @ A

            Gxx, Gxu, Gux, Guu = self.regularize(Gxx, Gxu, Gux, Guu)
        
            d[k] = np.linalg.pinv(Guu) @ gu
            K[:, :, k] = np.linalg.pinv(Guu) @ Gux

            #update p and P
            
            gradJ += gu.T @ d[k] 
        return gradJ
    
    def control(self, x0, xgoal, xtraj, utraj):
        self.xgoal = xgoal
        p = None
        P= None
        d= None
        K= None
        gradJ= None
        xn= None
        un= None
        gx= None
        gu= None
        Gxx= None
        Guu= None
        Gxu= None
        Gux= None
        while np.maximum(np.abs(d)) > 1e-3:

            gradJ = self.backward_pass_(xn, un, p P, d, K)

            xn[:, 1] = xtraj[:, 1]
            alpha = 1.0

            for k in range(1, self.N_-1):
                un[:, k] = utraj[:, k] - alpha*d[k] - np.dot(K[:, :, k], xn[:, k] - xtraj[:, k])
                xn[:, k+1] = self.f_(xn[:, k], un[:, k])
            Jn = self.cost_(xn, un)

            #Conduct line search on Jn

            J = Jn
            xtraj = xn
            utraj = un
        
        return xtraj, utraj