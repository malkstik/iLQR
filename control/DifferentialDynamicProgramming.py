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
        self.N = int(self.Tf_/self.h_) + 1
        self.thist = np.arange(0, self.N-1, self.h_)

        self.Q_ = Q
        self.R_ = R
        self.Qf_ = Qf

        self.f_ = lambda x, u: RK4(dynamics, x, u, h = self.h_)

        self.dfdx_ = jax.grad(self.f_, argnums=0)
        self.dfdu_ = jax.grad(self.f_, argnums=1)

        self.dAdx_ = jax.grad(self.dfdx_, argnums=0)
        self.dBdx_ = jax.grad(self.dfdu_, argnums=0)
        self.dAdu_ = jax.grad(self.dfdx_, argnums=1)
        self.dBdu_ = jax.grad(self.dfdu_, argnums=1)

        self.xgoal = None

    def stage_cost_(self, x: np.ndarray, u: np.ndarray):
        return 0.5*((x - self.xgoal).T @ self.Q_ @ (x - self.xgoal)) + 0.5* u.T*self.R_*u

    def terminal_cost_(self, xf):
        return xf.T@self.Qf_@xf

    def cost_(self, xtraj, utraj):
        J = 0
        xf = xtraj[:, -1]
        for k in range(self.N-1):
            J += self.stage_cost_(xtraj[:,:-1], utraj)
        J += self.terminal_cost_(xf)

        return J

    def full_hessian_(self, Gxx, Gxu, Gux, Guu)
        G = np.vstack( 
            (np.hstack((Gxx, Gxu)),
                np.hstack((Gux, Guu)))
            )
        return G
    
    def backward_pass_(self, xtraj, utraj, p, P, d, K):
        deltaJ = 0

        p[:, self.N] = self.Qn_*(xtraj[:, -1] - self.xgoal)
        P[:, :, self.N] = self.Qn_

        for k in range(self.N)[::-1]:
            xk, uk = xtraj[:, k], utraj[:, k] 
            q = self.Q_ @ (xk-self.xgoal)
            r = self.R_ @ uk

            # Linearization
            A = self.dfdx_(xk, uk)
            B = self.dfxu_(xk, uk)

            # Action value gradient terms
            gx = q + A.T @ p[:, k+1]
            gu = r + B.T @ p[:, k+1]

            # Action value hessian terms
            Gxx = self.Q_ + A.T @ P[:, :, k+1] @ A
            Guu = self.R_ + B.T @ P[:, :, k+1] @ B
            Gxu = A.T @ P[:, :, k+1] @ B
            Gux = B.T @ P[:, :, k+1] @ A

            #Regularize
            beta = 0.1
            while not is_pos_def(self.full_hessian_(Gxx, Gxu, Gux, Guu)):
                Gxx += beta * A.T @ A
                Guu += beta * B.T @ B
                Gxu += beta * A.T @ B
                Gux += beta * B.T @ A
                beta *= 2
                
            # Feedforward term
            d[k] = np.linalg.inv(Guu) @ gu
            # Feedback term
            K[:, :, k] = np.linalg.inv(Guu) @ Gux

            #update p and P
            p[:, k] = (gx - K[:, :, k].T @ gu + K[:, :, k].T @ Guu @ d[k] - Gxu @ d[k]).squeeze()
            P[:, :, k] = Gxx + K[:, :, k].T @ Guu @ K[:, :, k] - K[:, :, k].T @ Gux
            
            deltaJ += gu.T @ d[k] 
        return deltaJ
    
    def forward_rollout(self, xtraj, utraj, d, K, alpha):
        xn = np.zeros(self.nx_, self.N)
        un = np.zeros(self.nu_, self.N-1)
        for k in range(1, self.N-1):
                un[:, k] = utraj[:, k] - alpha*d[k] - np.dot(K[:, :, k], xn[:, k] - xtraj[:, k])
                xn[:, k+1] = self.f_(xn[:, k], un[:, k])
        Jn = self.cost_(xn, un)

        return xn, un, Jn

    def control(self, x0, xgoal, xtraj, utraj):
        self.xgoal = xgoal
        p = None        #gradients
        P= None         #hessians 
        d= None         #feedforward
        K= None         #feedback
        deltaJ= None    #change to trajectory cost 
        xn= None        #state traj
        un= None        #action traj
        gx= None        #gradient x term
        gu= None        #gradient u term
        Gxx= None       #Hessian xx term
        Guu= None       #Hessian uu term
        Gxu= None       #Hessian xu term
        Gux= None       #Hessian ux term


        #Initial rollout
        for k in range (1, self.N-1):
            xtraj[:, k+1] = self.dynamics_(xtraj[:, k], utraj[:, k])
        J = self.cost_(xtraj, utraj)

        while np.maximum(np.abs(d)) > 1e-3:

            deltaJ = self.backward_pass_(xn, un, p P, d, K)

            # Forward rollout with line search
            xn[:, 1] = xtraj[:, 1]
            alpha = 1.0

            xn, un, Jn = self.forward_rollout(self, xtraj, utraj, d, K, alpha)
            while np.isnan(Jn) or Jn > (J - 1e-2 * alpha * deltaJ):
                xn, un, Jn = self.forward_rollout(self, xtraj, utraj, d, K, alpha)

            J = Jn
            xtraj = xn
            utraj = un
        
        return xtraj, utraj
    
if __name__ == "__main__":
    def dynamics():
        pass
    
    controller_params = {"dynamics": dynamics,
                         "nx": 4,
                         "nu": 1,
                         "Q":  np.diag([1.0, 1.0, 1.0, 1.0]),
                         "R":  np.diag([0.1]),
                         "Qf": np.array([100.0, 100.0, 100.0, 100.0])
                         }
    controller = iLQR(**controller_params)
    
    x0 = np.zeros((4,1))
    xgoal = np.array(([0.0, np.pi, 0.0, 0.0])).reshape((4,1))
    xtraj = np.kron(np.ones((1, controller.N)), x0)
    utraj = 0.001 * np.ones(((controller.nu_, controller.N)))
    xtraj, utraj = controller.contol(x0, xgoal, xtraj, utraj)
