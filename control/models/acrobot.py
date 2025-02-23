from control.models.model import Model
import casadi as ca
import numpy as np

class Acrobot2DModel(Model):
    def __init__(self, Q, Qf, R, dt, N, 
                 integration = "RK4", **kwargs):

        self.Q = Q
        self.Qf = Qf
        self.R = R
        self.dt = dt
        self.N = N
        self.nx = 4
        self.nu = 1

        self.xref = np.zeros(self.nx, dtype = np.float64)
        self.uref = np.zeros(self.nu, dtype = np.float64)
    

        # Dynamics definition
        self.m1 = 1
        self.m2 = 1
        self.l1 = 1
        self.l2 = 2
        self.lc1 = 0.5
        self.lc2 = 1
        self.Ic1 = 0.083
        self.Ic2 = 0.33
        self.b1 = 0.1
        self.b2 = 0.1
        g = 9.81

        x_sym = ca.MX.sym('state', 4)  # State [th1, th2, th1_dot, th2_dot]
        u_sym = ca.MX.sym('action', 1)   # Control inputs (motor currents)

        th1, th2, th1_dot, th2_dot = x_sym[0], x_sym[1], x_sym[2], x_sym[3]
        s1, s2, s12 = ca.sin(th1), ca.sin(th2), ca.sin(th1 + th2)
        c2 = ca.cos(th2)
        m2l1lc2 = self.m2*self.l1*self.lc2

        I1 = self.Ic1 + self.m1*self.lc1**2
        I2 = self.Ic2 + self.m2*self.lc2**2
        
        M11 = I1 + I2 + self.m2*self.l1**2 + 2*m2l1lc2*c2
        M12 = I2 + m2l1lc2*c2
        M22 = I2
        M = ca.vertcat(
            ca.horzcat(M11, M12),
            ca.horzcat(M12, M22)  
        )
        
        Coreolis1 = -2*m2l1lc2*s2*th2_dot*th1_dot - m2l1lc2*s2*th2_dot**2
        Coreolis2 = m2l1lc2*s2*th1_dot**2

        Grav1 = g*self.m1*self.lc1*s1 + g*self.m2*(self.l1*s1+self.lc2*s12)
        Grav2 = g*self.m2*self.lc2*s12

        Damping1 = self.b1*th1_dot
        Damping2 = self.b2*th2_dot
                
        Bias = ca.vertcat(Coreolis1 + Grav1 + Damping1, Coreolis2 + Grav2 + Damping2)

        B = np.array([[0],
                      [1]])
        
        x_dot = ca.vertcat(
            th1_dot,
            th2_dot,
            ca.solve(M, B@u_sym-Bias) #th1_ddot, th2_ddot
        )
        self.xdot_fun = ca.Function('xdot', [x_sym, u_sym], [x_dot])

        # RK4 integration
        if integration == "RK4":
            print("Integrating dynamics with RK4")
            f1 = self.xdot_fun(x_sym.T, u_sym).T
            f2 = self.xdot_fun(x_sym.T + 0.5*f1*dt, u_sym).T
            f3 = self.xdot_fun(x_sym.T + 0.5*f2*dt, u_sym).T
            f4 = self.xdot_fun(x_sym.T + 1.0*f3*dt, u_sym).T
            x_next = x_sym.T + (f1 + 2*f2 + 2*f3 + f4)*dt/6.0
        # Euler integration
        else:
            print("Integrating dynamics with Euler")
            x_next = x_sym.T + self.xdot_fun(x_sym.T, u_sym).T * dt

        self.x_next = ca.Function('xnext', [x_sym, u_sym], [x_next])
        # Compute Jacobians (A and B matrices)
        A_continuous = ca.jacobian(x_dot, x_sym)
        B_continuous = ca.jacobian(x_dot, u_sym)
        A_discrete = ca.jacobian(x_next, x_sym)  
        B_discrete = ca.jacobian(x_next, u_sym)  

        # CasADi function for A and B matrices
        self.A_discrete_fun = ca.Function('A_discrete', [x_sym, u_sym], [A_discrete])
        self.B_discrete_fun = ca.Function('B_discrete', [x_sym, u_sym], [B_discrete])      
        self.A_continuous_fun = ca.Function('A_continuous', [x_sym, u_sym], [A_continuous])
        self.B_continuous_fun = ca.Function('B_continuous', [x_sym, u_sym], [B_continuous])      
        
    def set_references(self, xref, uref):
        self.xref = xref
        self.uref = uref

    def discrete_dynamics(self, x, u):   
        x_next = self.x_next(x, u)
        return x_next

    def _cost_stage(self, x: np.ndarray, u: np.ndarray, xref: np.ndarray, uref: np.ndarray):
        '''
        Computes cost due to a state action tuple

        :param x: state trajectory, ndarray of shape (N-1, nx)
        :param u: action trajectory, ndarray of shape (N-1, nu)
        '''        
        xerr = (x - xref).reshape((4,1))
        uerr = u - uref

        state_cost = 0.5*(xerr).T @ self.Q @ (xerr) 
        action_cost = 0.5*(uerr).T @ self.R @ (uerr)
        return state_cost + action_cost

    def _cost_final(self, xf: np.ndarray, xref: np.ndarray):
        '''
        Computes cost due to final state of state trajectory
        '''
        return 0.5*(xf - xref).T @ self.Qf @ (xf - xref)
    
    def cost_trj(self, xtraj: np.ndarray, utraj: np.ndarray):
        '''
        Computes total cost of trajectory
        :param xtraj: state trajectory, ndarray of shape (num_time_steps, nx)
        :param utraj: action trajectory, ndarray of shape (num_time_steps-1, nu)

        '''
        J = 0
        xf = xtraj[-1, :]

        for n in range(self.N-1):
            J+= self._cost_stage(xtraj[n,:], utraj[n,:], self.xref, self.uref)
        J += self._cost_final(xf, self.xref)
        return J

    def _linearize_discrete(self, ref_state, ref_action):
        discreteA, discreteB = self.A_discrete_fun(ref_state, ref_action), self.B_discrete_fun(ref_state, ref_action) 
        return discreteA, discreteB

    def _linearize_continuous(self,ref_state, ref_action):
        contA, contB = self.A_continuous_fun(ref_state, ref_action), self.B_continuous_fun(ref_state, ref_action) 
        return contA, contB
    
    def stage(self, x, u):         
        f_x, f_u = self._linearize_discrete(x, u)

        l_x = self.Q @ (x - self.xref)
        l_u = self.R @ (u - self.uref)
        l_xx = self.Q
        l_uu = self.R
        l_ux = np.zeros((self.nu, self.nx))    

        return l_x, l_u, l_xx, l_ux, l_uu, f_x, f_u 
    
    def final(self, x):
        V_xx = self.Qf
        V_x = self.Qf @ (x - self.xref)
        return V_x, V_xx