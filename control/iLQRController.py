import numpy as np
from maths import linalg
from control.models.model import Model
import sys
from time import time
from utils.timeit import timeit

class iLQR:
    def __init__(self, 
                model: Model,
                N: int,         
                max_iter: int,
                regu_init: int,
                min_regu: float = 0.001,
                max_regu: float = 10000,
                max_linesearch_iters: int = 20,     
                **kwargs):
        self.model = model
        self.nx, self.nu = self.model.get_dims()
        self.N = N
        self.max_iter = max_iter
        self.regu_init = regu_init
        self.min_regu = min_regu
        self.max_regu = max_regu
        self.max_linesearch_iters = max_linesearch_iters    

    def run_ilqr(self, x0, u_trj):
        x_trj = self.model.rollout(x0, u_trj)
        total_cost = self.model.cost_trj(x_trj, u_trj)

        J = total_cost      
        Jprev = sys.float_info.max

        d = np.zeros([u_trj.shape[0], u_trj.shape[1]])
        K = np.zeros([u_trj.shape[0], u_trj.shape[1], x_trj.shape[1]])
        
        self.regu = self.regu_init

        iters = 0
        while abs(J - Jprev) > 1e-6 and iters < self.max_iter:
            iters += 1
            Jprev = J

            # start = time()
            Jn = 0.0
            d[:], K[:], deltaJ = self._backward_pass(x_trj, u_trj, d, K)
            # print(f"backward takes {time() - start}")
            alpha = 1
            # start = time()
            x_trj_new, u_trj_new = self._forward_pass(x_trj, u_trj, d, K, alpha)
            # print(f"forwardtakes {time()-start}")

            line_search_iters = 0

            complete_line_search = False
            abandon_line_search = False

            # start = time()
            while not complete_line_search and not abandon_line_search: 
                complete_line_search = Jn > (J - 1e-2 * alpha * deltaJ) or 1e-2 * alpha * deltaJ >1e-4
                abandon_line_search = np.isnan(Jn) or line_search_iters > self.max_linesearch_iters

                line_search_iters += 1
                alpha *= 0.5

                # New rollout for reduce step
                x_trj_new, u_trj_new = self._forward_pass(x_trj, u_trj, d, K, alpha)
                Jn = self.model.cost_trj(x_trj_new, u_trj_new)

                # print(Jn-J)


            if complete_line_search:
                x_trj[:] = x_trj_new
                u_trj[:] = u_trj_new
            
            if abandon_line_search:
                self.regu *= 1.6

            self.regu = min(max(self.regu, self.min_regu), self.max_regu)
            J = Jn     
            return x_trj, u_trj    

  

    def _forward_pass(self, x_trj, u_trj, k_trj, K_trj, alpha = 1):
        x_trj_new = np.zeros(x_trj.shape)
        x_trj_new[0, :] = x_trj[0, :]
        u_trj_new = np.zeros(u_trj.shape)

        for n in range(u_trj.shape[0]):
            u_trj_new[n,:] = u_trj[n] + alpha*k_trj[n] + K_trj[n].dot(x_trj_new[n] - x_trj[n])
            x_trj_new[n+1,:] = self.model.discrete_dynamics(x_trj_new[n,:], u_trj_new[n,:])
        return x_trj_new, u_trj_new

    def _backward_pass(self, x_trj, u_trj, d, K):
        expected_cost_redu = 0

        V_x, V_xx = self.model.final(x_trj[-1])
        for k in range(u_trj.shape[0] - 1, -1, -1):

            start = time()
            l_x, l_u, l_xx, l_ux, l_uu, f_x, f_u = self.model.stage(x_trj[k], u_trj[k])
            print(f"autodiff took {time()-start}")
            Q_x, Q_u, Q_xx, Q_ux, Q_uu = self._Q_terms(l_x, l_u, l_xx, l_ux, l_uu, f_x, f_u, V_x, V_xx)

            # We add regularization to ensure that Q_uu is invertible and nicely conditioned
            Q_uu_regu = Q_uu + np.eye(Q_uu.shape[0]) * self.regu
            while linalg.is_singular(Q_uu_regu):
                self.regu *= 1.6
                Q_uu_regu += self.regu*np.eye(self.nu)     

            d_k, K_k = self._gains(Q_uu_regu, Q_u, Q_ux)
            d[k, :] = d_k
            K[k, :, :] = K_k
            V_x, V_xx = self._V_terms(Q_x, Q_u, Q_xx, Q_ux, Q_uu, K_k, d_k)
            expected_cost_redu += self._expected_cost_reduction(Q_u, Q_uu, d_k)
        self.regu /= 1.6
        return d, K, expected_cost_redu  
    
    def _Q_terms(self, l_x, l_u, l_xx, l_ux, l_uu, f_x, f_u, V_x, V_xx):
        Q_xx = l_xx + f_x.T @ V_xx @ f_x
        Q_x = l_x + f_x.T @ V_x
        Q_u = l_u + f_u.T @ V_x

        Q_ux = l_ux + f_u.T @ V_xx @ f_x
        Q_uu = l_uu + f_u.T @ V_xx @ f_u
        return Q_x, Q_u, Q_xx, Q_ux, Q_uu
    
    def _gains(self, Q_uu, Q_u, Q_ux):
        # Q_uu_inv = linalg.qr_inverse(Q_uu)
        Q_uu_inv = np.linalg.inv(Q_uu)
        k = -Q_uu_inv @ Q_u
        K = -Q_uu_inv@ Q_ux
        return k, K
    
    def _V_terms(self, Q_x, Q_u, Q_xx, Q_ux, Q_uu, K, k):
        V_xx = Q_xx + K.T @ Q_uu @ K + K.T @ Q_ux + Q_ux.T @ K
        V_x = Q_x + K.T @ Q_uu @ k + K.T @ Q_u + Q_ux.T @ k

        return V_x, V_xx
    
    def _expected_cost_reduction(self, Q_u, Q_uu, k):
        return -Q_u.T.dot(k) - 0.5 * k.T.dot(Q_uu.dot(k))
