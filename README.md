# Description
This repository presents an implementation of iterative Linear Quadratic Regulator (iLQR) using pydrake and CasADi. Included are examples using an acrobot, quadcopter with euler angle representation, and quadcopter with quaternion representation.

# Examples
## Acrobot
![Demo GIF](https://github.com/malkstik/iLQR/blob/main/results/AcrobotiLQR.gif)
## Quadrotor using Quaternions
![Demo GIF](https://github.com/malkstik/iLQR/blob/main/results/QuadcopteriLQR.gif)

# iLQR Controller

This repository provides an implementation of the **Iterative Linear Quadratic Regulator (iLQR)** algorithm for trajectory optimization and control. It is designed to work with custom dynamics models, cost functions, and derivatives, allowing you to apply it to various types of control problems.

#Usage
##Creating a Custom Model

The iLQR controller requires a model class that describes the dynamics, cost function, and derivatives of the system you're controlling. To use the controller, you need to define a child class of Model and implement the following methods:

    model.discrete_dynamics(x, u): Returns the system's next state, given the current state x and control input u.
    model._cost_stage(x, u): Returns the cost for a given state x and control input u.
    model._cost_final(x): Returns the terminal cost for a given state x
    model.stage(x, u): Returns the relevant derivatives of stage cost along linearization of discrete dynamics for a given state x and control input u
    model.final(x): Returns the derivatives of the terminal cost at the final state

Here’s a basic example for defining a custom model:

```
class CarModel(Model):
    def __init__(self, r, v_target):
        self.nx = 5
        self.nu = 2
        self.ndx = 5

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
```
## Setting Up the Controller

Once you have created your custom model, you can initialize the iLQR controller. The controller takes the model and a dictionary of parameters as inputs. Here's how you would set it up:
```python
iLQRParams = {
    "N": 40,                           # Number of time steps in the trajectory
    "max_iter": 50,                     # Maximum number of iterations for iLQR
    "max_linesearch_iters": 20          # Maximum number of line search iterations
}
# Define your custom model (e.g., CarModel)
r = 2
v_target = 2
plant = CarModel(r, v_target)

# Initialize the iLQR controller
controller = iLQR(plant, **iLQRParams)
```
Running the Controller

After initializing the controller, you can use it to compute the optimal control trajectory. The exact interface depends on your model and setup, but typically, you would call the run method of the controller with an initial trajectory and other necessary parameters.

For example:

## Define the initial trajectory guess (this will depend on your problem)
x_init = np.zeros((state_dim, N))  # Initial state trajectory (e.g., car states)
u_init = np.zeros((control_dim, N-1))  # Initial control input trajectory

## Run iLQR to optimize the trajectory
optimal_trajectory, optimal_controls = controller.run(x_init, u_init)

The run method will return the optimized state trajectory (optimal_trajectory) and control inputs (optimal_controls) that minimize the cost.
Parameters

The iLQR controller takes several parameters for fine-tuning:

    N: The number of time steps in the trajectory to optimize.
    max_iter: The maximum number of iterations for the iLQR algorithm.
    max_linesearch_iters: The maximum number of iterations for line search used to adjust regularization.
    d_tol: Convergence tolerance for infinity norm of feedforward gain

Example

Here’s a full example of how to use the iLQR controller:
```python
import numpy as np
from control.iLQRController import iLQR
from control.models.car import CarModel

iLQRParams = {"N": 40,
              "max_iter": 50,
              "max_linesearch_iters": 20
}
r = 2
v_target = 2
u_init_trj = np.random.randn(iLQRParams['N'] - 1, 2) * 0.0001

plant = CarModel(r, v_target)
controller = iLQR(plant, **iLQRParams)

x_trj = x0
u_trj = None
for n in range(50):
    x_ctrl, u_init_trj= controller.run_ilqr(x0, u_init_trj)
    x0 = x_ctrl[1,:]
    u_ctrl = u_init_trj[0, :]
    x_trj = np.vstack((x_trj, x_ctrl[1,:]))
    u_trj = np.vstack((u_trj, u_ctrl)) if u_trj is not None else u_ctrl
```
