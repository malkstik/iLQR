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
                max_linesearch_iters: int = 20,     
                d_tol: float = 1e-3,
                **kwargs):
        self.model = model
        self.nx, self.nu = self.model.get_dims()
        self.N = N
        self.max_iter = max_iter
        self.max_linesearch_iters = max_linesearch_iters    
        self.d_tol = d_tol

        self.num_ls_success = 0
        self.num_ls_fails = 0
        self.num_iters = 0

    def run_ilqr(self, x0, u_trj):
        x_trj = self.model.rollout(x0, u_trj)
        if np.any(np.isnan(x_trj)):
            print(x_trj)    
            print(u_trj)  

        total_cost = self.model.cost_trj(x_trj, u_trj)

        J = total_cost      
        Jprev = J

        d = sys.float_info.max*np.ones([self.N-1, self.nu])
        K = np.zeros([self.N-1, self.nu, self.nx])
        
        self.num_iters += 1
        iters = 0

        while np.max(np.linalg.norm(d, axis=1)) > self.d_tol and iters < self.max_iter:
        # while abs(J - Jprev) > 5e-4 and iters < self.max_iter or iters==0:
        # while iters < self.max_iter:
            iters += 1
            Jprev = J

            d[:], K[:], deltaJ = self._backward_pass(x_trj, u_trj, d, K)
            if deltaJ < 1:
                break
            #Line search
            line_search_iters = 0
            alpha = 1
            complete_line_search = False
            abandon_line_search = False
            self.num_ls_success = 0
            self.num_ls_fails = 0
            while not complete_line_search and not abandon_line_search: 
                line_search_iters += 1
                # New rollout for reduce step
                x_trj_new, u_trj_new = self._forward_pass(x_trj, u_trj, d, K, alpha)
                Jn = self.model.cost_trj(x_trj_new, u_trj_new)

                complete_line_search = Jn - J <= -1e-2*alpha * deltaJ and line_search_iters < self.max_linesearch_iters 
                abandon_line_search = np.isnan(Jn) or line_search_iters >= self.max_linesearch_iters 
                alpha *= 0.5
            if complete_line_search:
                self.num_ls_success += 1
                x_trj[:] = x_trj_new
                u_trj[:] = u_trj_new 
            if abandon_line_search:
                self.num_ls_fails += 1
            # print(f"Iters: {line_search_iters}, Successful LS: {self.num_ls_success}, Failed LS: {self.num_ls_fails}" )
            # print(f"Improvement: {Jn - J}")
            # print(f"Expected Improvement: {-deltaJ}")
            # print()
            J = Jn    

        # print(f"Iters before convergence: {iters}") 
        # print(f"%Cost reduction: {(J - Jprev)/Jprev}")
        # print(f"Biggest d: {np.max(np.linalg.norm(d, axis=1))}")
        return x_trj, u_trj

  
    def _forward_pass(self, x_trj, u_trj, d_trj, K_trj, alpha = 1):
        x_trj_new = np.zeros(x_trj.shape)
        x_trj_new[0, :] = x_trj[0, :]
        u_trj_new = np.zeros(u_trj.shape)

        for n in range(self.N-1):
            u_trj_new[n,:] = u_trj[n] + alpha*d_trj[n] + K_trj[n].dot(x_trj_new[n] - x_trj[n])
            x_trj_new[n+1,:] = self.model.discrete_dynamics(x_trj_new[n,:], u_trj_new[n,:])
        return x_trj_new, u_trj_new
    
    def _backward_pass(self, x_trj, u_trj, d, K):
        expected_cost_redu = 0

        V_x, V_xx = self.model.final(x_trj[-1, :])
        for k in range(self.N-2, -1, -1):
            l_x, l_u, l_xx, l_ux, l_uu, f_x, f_u = self.model.stage(x_trj[k], u_trj[k])
            Q_x, Q_u, Q_xx, Q_ux, Q_uu = self._Q_terms(l_x, l_u, l_xx, l_ux, l_uu, f_x, f_u, V_x, V_xx)

            # We add regularization to ensure that Q_uu is invertible and nicely conditioned
            regu = 0.1
            Q = self._full_hessian(Q_xx, Q_ux, Q_uu)
            if np.any(np.isnan(Q)):
                print(x_trj[k], u_trj[k])
            while np.any(np.linalg.eigvals(Q) < 0):
                Q_xx += regu* f_x.T @ f_x
                Q_uu += regu* f_u.T @ f_u
                Q_ux += regu* f_u.T @ f_x    
                Q[:] = self._full_hessian(Q_xx, Q_ux, Q_uu)
                regu *= 2   
            # print(f"Necessary regularization: {regu}")
            d_k, K_k = self._gains(Q_uu, Q_u, Q_ux)
            d[k, :] = d_k
            K[k, :, :] = K_k
            V_x, V_xx = self._V_terms(Q_x, Q_u, Q_xx, Q_ux, Q_uu, K_k, d_k)

            expected_cost_redu += self._expected_cost_reduction(Q_u, Q_uu, d_k)
        return d, K, expected_cost_redu  

    def _Q_terms(self, l_x, l_u, l_xx, l_ux, l_uu, f_x, f_u, V_x, V_xx):
        Q_x = l_x + f_x.T @ V_x
        Q_u = l_u + f_u.T @ V_x

        Q_xx = l_xx + f_x.T @ V_xx @ f_x
        Q_ux = l_ux + f_u.T @ V_xx @ f_x
        Q_uu = l_uu + f_u.T @ V_xx @ f_u
        return Q_x, Q_u, Q_xx, Q_ux, Q_uu
    
    def _gains(self, Q_uu, Q_u, Q_ux):
        Q_uu_inv = np.linalg.inv(Q_uu)
        d = -Q_uu_inv @ Q_u
        K = -Q_uu_inv@ Q_ux
        return d, K
    
    def _V_terms(self, Q_x, Q_u, Q_xx, Q_ux, Q_uu, K, d):
        V_xx = Q_xx + K.T @ Q_uu @ K + K.T @ Q_ux + Q_ux.T @ K
        V_x = Q_x + K.T @ Q_uu @ d + K.T @ Q_u + Q_ux.T @ d
        return V_x, V_xx
    
    def _expected_cost_reduction(self, Q_u, Q_uu, d):
        # return -Q_u @ d

        return -Q_u @ d - 0.5 * d.T @ (Q_uu.T @ d)

    def _full_hessian(self, Q_xx: np.ndarray, Q_ux: np.ndarray, Q_uu: np.ndarray)-> np.ndarray:
        '''
        Combines hessian parts into one
        '''
        G = np.vstack( 
            (np.hstack((Q_xx, Q_ux.T)),
             np.hstack((Q_ux, Q_uu)))
            )
        return G
