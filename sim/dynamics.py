from abc import ABC, abstractmethod


from sim_utils import RK4
import numpy as np
import matplotlib
# matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class BaseDynamics(ABC):    
    def __init__(self):
        self.nx: int = None
        self.nu: int = None

    @abstractmethod
    def continuous_dynamics(self, x, u):
        raise NotImplementedError

    @abstractmethod
    def animate(self, x0, utraj, dt):
        raise NotImplementedError

    def sim(self, x0: np.array, utraj: np.array, dt: float):
        '''
        Rollout actions given initial statae
        :param utraj: array of shape (horizon, nu)
        :param dt: fixed time step for RK4 solver
        :param x0: initial state of shape (nx, 1), will default to current state if none is provided
        :return xtraj: array of shape (horizon, nx) containing the states from rollout
        '''
        
        x = self.state if x0 is None else x0
        xtraj = x.copy()
        utraj = np.expand_dims(utraj, -1)

        for action in utraj:
            x = RK4(self.continuous_dynamics, x, action, dt)
            xtraj = np.hstack((xtraj, x))
        
        xtraj = xtraj.reshape((-1, self.nx))

        return xtraj
    
    def plot(self):
        # Subplots of each state
        pass
    
class CartPole(BaseDynamics):
    def __init__(self):
        pass

    def continuous_dynamics(self, x, u):
        return super().continuous_dynamics(x, u)
    


class Acrobot(BaseDynamics):
    def __init__(self, m1: float, m2: float, l1: float, l2: float):
        self.nx = 4
        self.nu = 1

        self.m1 = m1
        self.m2 = m2
        self.l1 = l1
        self.l2 = l2
        
        self.lc1 = l1/2
        self.lc2 = l2/2

        self.I1 = 1/12 * self.m1 * l1**2 + self.m1 * self.lc1**2
        self.I2 = 1/12 * self.m2 * l2**2 + self.m2 * self.lc2**2

        self.state = np.zeros((4,1))
        self.B = np.array([[0.0],
                           [1.0]])
        
    def update_manipulator_equation(self, x: np.array):
        '''
        updates state and matrices of the manipulator equation
        :param x: state to update with
        :return (M, C, tau): tuple of inertial, coriolis, and forcing matrices
        '''
        self.state = x

        theta1, theta2 = self.state[0,0], self.state[1,0]
        thetad1, thetad2 = self.state[2,0], self.state[3,0]
        s1 = np.sin(theta1)
        s2, c2 = np.sin(theta2), np.cos(theta2)
        s12 = np.sin(theta1 + theta2)
        g = 9.81

        self.M = np.array([[self.I1 + self.I2 + self.m2*self.l1**2 + 2*self.m2*self.l1*self.lc2*c2, self.I2 + self.m2*self.l1*self.lc2*c2],
                          [self.I2 + self.m2*self.l1*self.lc2*c2, self.I2]])
        
        self.C = np.array([[-2*self.m2*self.l1*self.lc2*s2*thetad2, -self.m2*self.l1*self.lc2*s2*thetad2],
                          [self.m2*self.l1*self.lc2*s2*thetad1, 0.0]])

        self.tau = np.array([[-self.m1*g*self.lc1*s1 - self.m2*g*(self.l1*s1 + self.lc2*s12)],
                             [-self.m2*g*self.lc2*s12]])
        return self.M, self.C, self.tau

    def get_link_positions(self):
        '''
        retrieve the CoM position of the links
        :return (pc1, pc2): returns tuple of base link and outer link, both as arrays of shape (2 ,1)
        '''
        theta1, theta2 = self.state[0,0], self.state[1,0]
        pc1 = np.array([self.lc1*np.sin(theta1),
                        -self.lc1*np.cos(theta1)])
        pc2 = pc1 + np.array([self.lc2*np.sin(theta1 + theta2),
                             -self.lc2*np.cos(theta1 + theta2)])
        
        return pc1, pc2

    def continuous_dynamics(self, x: np.array, u: np.array):
        '''
        compute time derivative of state
        :param x: array of shape (nx, 1) containing state
        :param u: array of shape (nu, 1) containing action
        :return xdot: array of shape (nx, 1) containing state time derivative
        '''
        q, qd = x[:2], x[2:]
        M, C, tau = self.update_manipulator_equation(x)

        xdot = np.zeros_like(self.state)
        xdot[:2] = q
        xdot[2:] = np.linalg.inv(M) @ (tau + self.B @ u - C @ qd)
        return xdot
    
    def animate(self, x0, utraj, dt):
        xtraj = self.sim(x0, utraj, dt)
        
        p1_traj = None
        p2_traj = None

        for x in xtraj:
            theta1, theta2 = x[0], x[1]
            p1 = np.array([[self.l1*np.sin(theta1)],
                            [-self.l1*np.cos(theta1)]])
            p2 = p1 + np.array([[self.l2*np.sin(theta1 + theta2)],
                                [-self.l2*np.cos(theta1 + theta2)]])
            
            p1_traj = p1 if p1_traj is None else np.hstack((p1_traj, p1))
            p2_traj = p2 if p2_traj is None else np.hstack((p2_traj, p2))


        fig, ax = plt.subplots()
        maxL = (self.l1+self.l2)
        ax.set_xlim(-1.5 * maxL, 1.5 * maxL)
        ax.set_ylim(-1.5 * maxL, 1.5 * maxL)
        acrobot, = ax.plot([], [], 'o-', lw=2)  # Line to animate

        def init():
            acrobot.set_data([], [])
            return acrobot,
    
        def animate(i):
            acrobot.set_data([0, p1_traj[0, i], p2_traj[0, i]], [0, p1_traj[1, i], p2_traj[1, i]])
            return acrobot,

        ani = FuncAnimation(fig, animate, init_func=init, frames=xtraj.shape[0], interval=dt*1000, blit=True)
        plt.show()
    

class QuadRotor2D(BaseDynamics):
    def __init__(self):
        pass

    def continuous_dynamics(self, x, u):
        return super().continuous_dynamics(x, u)



if __name__ == "__main__":
    plant = Acrobot(1.0, 1.0, 1.0, 1.0)
    x0 = np.array([0, 0, 0, 0]).reshape((4,1))
    utraj = 0.005*np.random.rand(600, 1)
    # utraj = 0.005*np.ones((30,1))
    plant.animate(x0, utraj, 0.02)