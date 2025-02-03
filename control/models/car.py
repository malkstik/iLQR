import pydrake.symbolic as sym
import numpy as np
from control.models.model import Model

class derivatives:
    def __init__(self, discrete_dynamics, cost_stage, cost_final, n_x, n_u):
        self.x_sym = np.array(
            [sym.Variable("x_{}".format(i)) for i in range(n_x)]
        )
        self.u_sym = np.array(
            [sym.Variable("u_{}".format(i)) for i in range(n_u)]
        )
        x = self.x_sym
        u = self.u_sym

        l = cost_stage(x, u)
        self.l_x = sym.Jacobian([l], x).ravel()
        self.l_u = sym.Jacobian([l], u).ravel()
        self.l_xx = sym.Jacobian(self.l_x, x)
        self.l_ux = sym.Jacobian(self.l_u, x)
        self.l_uu = sym.Jacobian(self.l_u, u)

        l_final = cost_final(x)
        self.l_final_x = sym.Jacobian([l_final], x).ravel()
        self.l_final_xx = sym.Jacobian(self.l_final_x, x)

        f = discrete_dynamics(x, u)
        self.f_x = sym.Jacobian(f, x)
        self.f_u = sym.Jacobian(f, u)

    def stage(self, x, u):
        env = {self.x_sym[i]: x[i] for i in range(x.shape[0])}
        env.update({self.u_sym[i]: u[i] for i in range(u.shape[0])})

        l_x = sym.Evaluate(self.l_x, env).ravel()
        l_u = sym.Evaluate(self.l_u, env).ravel()
        l_xx = sym.Evaluate(self.l_xx, env)
        l_ux = sym.Evaluate(self.l_ux, env)
        l_uu = sym.Evaluate(self.l_uu, env)

        f_x = sym.Evaluate(self.f_x, env)
        f_u = sym.Evaluate(self.f_u, env)

        return l_x, l_u, l_xx, l_ux, l_uu, f_x, f_u

    def final(self, x):
        env = {self.x_sym[i]: x[i] for i in range(x.shape[0])}

        l_final_x = sym.Evaluate(self.l_final_x, env).ravel()
        l_final_xx = sym.Evaluate(self.l_final_xx, env)

        return l_final_x, l_final_xx

class CarModel(Model):
    def __init__(self, r, v_target):
        self.nx = 5
        self.nu = 2

        self.r = r
        self.v_target = v_target
        self.derivs = derivatives(self.discrete_dynamics, self._cost_stage, self._cost_final, self.nx, self.nu)

    def _car_continuous_dynamics(self, x, u):
        # x = [x position, y position, heading, speed, steering angle]
        # u = [acceleration, steering velocity]
        m = sym if x.dtype == object else np  # Check type for autodiff
        heading = x[2]
        v = x[3]
        steer = x[4]
        x_d = np.array(
            [v * m.cos(heading), v * m.sin(heading), v * m.tan(steer), u[0], u[1]]
        )
        return x_d

    def discrete_dynamics(self, x, u):
        dt = 0.1
        x_next = self._car_continuous_dynamics(x, u) * dt + x
        return x_next 
    
    def _cost_stage(self, x, u):
        eps = 1e-6 
        m = sym if x.dtype == object else np  # Check type for autodiff
        c_circle = (m.sqrt(x[0] ** 2 + x[1] ** 2 + eps) - self.r) ** 2
        c_speed = (x[3] - self.v_target) ** 2
        c_control = (u[0] ** 2 + u[1] ** 2) * 0.1
        return c_circle + c_speed + c_control

    def _cost_final(self, x):
        eps = 1e-6         
        m = sym if x.dtype == object else np  # Check type for autodiff
        c_circle = (m.sqrt(x[0] ** 2 + x[1] ** 2 + eps) - self.r) ** 2
        c_speed = (x[3] - self.v_target) ** 2
        return c_circle + c_speed

    def stage(self, x, u):
        l_x, l_u, l_xx, l_ux, l_uu, f_x, f_u = self.derivs.stage(x, u)
        return l_x, l_u, l_xx, l_ux, l_uu, f_x, f_u 
    
    def final(self, x):
        V_x, V_xx = self.derivs.final(x)
        return V_x, V_xx