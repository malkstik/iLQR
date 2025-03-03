# Description
This repository presents an implementation of iterative Linear Quadratic Regulator (iLQR) using pydrake and CasADi. 

# Examples

## Acrobot
![Demo GIF](https://github.com/malkstik/iLQR/blob/main/results/AcrobotiLQR.gif)
## Quadrotor using Quaternions
![Demo GIF](https://github.com/malkstik/iLQR/blob/main/results/https://github.com/malkstik/iLQR/blob/main/results/acrobot_exact_dynamics.gif)

# iLQR Controller

This repository provides an implementation of the **Iterative Linear Quadratic Regulator (iLQR)** algorithm for trajectory optimization and control. It is designed to work with custom dynamics models, cost functions, and derivatives, allowing you to apply it to various types of control problems.

## Table of Contents
- [Usage](#usage)
  - [Creating a Custom Model](#creating-a-custom-model)
  - [Setting Up the Controller](#setting-up-the-controller)
  - [Running the Controller](#running-the-controller)
- [Parameters](#parameters)
- [Example](#example)

Usage
Creating a Custom Model

The iLQR controller requires a model class that describes the dynamics, cost function, and derivatives of the system you're controlling. To use the controller, you need to define a child class of Model and implement the following methods:

    model.dynamics(x, u): Returns the system's next state, given the current state x and control input u.
    model.cost(x, u): Returns the cost for a given state x and control input u.
    model.dynamics_derivative(x, u): Returns the derivative of the dynamics with respect to the state and control input.
    model.cost_derivative(x, u): Returns the derivative of the cost function with respect to the state and control input.

Here’s a basic example for defining a custom model:

```
class CarModel(Model):
    def dynamics(self, x, u):
        # Implement the car model dynamics
        # Example: return next state x_next based on current state x and control u
        pass

    def cost(self, x, u):
        # Implement the cost function based on state x and control u
        pass

    def dynamics_derivative(self, x, u):
        # Implement the derivative of the dynamics w.r.t. state and control
        pass

    def cost_derivative(self, x, u):
        # Implement the derivative of the cost w.r.t. state and control
        pass
```
Setting Up the Controller

Once you have created your custom model, you can initialize the iLQR controller. The controller takes the model and a dictionary of parameters as inputs. Here's how you would set it up:
```python
iLQRParams = {
    "N": 40,                           # Number of time steps in the trajectory
    "max_iter": 50,                     # Maximum number of iterations for iLQR
    "regu_init": 100,                   # Initial regularization parameter
    "min_regu": 0.001,                  # Minimum regularization value
    "max_regu": 10000,                  # Maximum regularization value
    "max_linesearch_iters": 20          # Maximum number of line search iterations
}
```
# Define your custom model (e.g., CarModel)
plant = CarModel()

# Initialize the iLQR controller
controller = iLQR(plant, **iLQRParams)

Running the Controller

After initializing the controller, you can use it to compute the optimal control trajectory. The exact interface depends on your model and setup, but typically, you would call the run method of the controller with an initial trajectory and other necessary parameters.

For example:

# Define the initial trajectory guess (this will depend on your problem)
x_init = np.zeros((state_dim, N))  # Initial state trajectory (e.g., car states)
u_init = np.zeros((control_dim, N-1))  # Initial control input trajectory

# Run iLQR to optimize the trajectory
optimal_trajectory, optimal_controls = controller.run(x_init, u_init)

The run method will return the optimized state trajectory (optimal_trajectory) and control inputs (optimal_controls) that minimize the cost.
Parameters

The iLQR controller takes several parameters for fine-tuning:

    N: The number of time steps in the trajectory to optimize.
    max_iter: The maximum number of iterations for the iLQR algorithm.
    max_linesearch_iters: The maximum number of iterations for line search used to adjust regularization.
    regu_init: The initial value for the regularization parameter.
    min_regu: The minimum regularization value.
    max_regu: The maximum regularization value.
    d_tol: Convergence tolerance for the difference in cost between iterations.

Example

Here’s a full example of how to use the iLQR controller:
```python
import numpy as np
from ilqr import iLQR
from car_model import CarModel  # Assume CarModel is defined in a separate file

# Define iLQR parameters
iLQRParams = {
    "N": 40,
    "max_iter": 50,
    "regu_init": 100,
    "min_regu": 0.001,
    "max_regu": 10000,
    "max_linesearch_iters": 20
}

# Initialize the model (e.g., CarModel)
plant = CarModel()

# Initialize the iLQR controller
controller = iLQR(plant, **iLQRParams)

# Define initial state and control trajectories
x_init = np.zeros((plant.state_dim, iLQRParams["N"]))
u_init = np.zeros((plant.control_dim, iLQRParams["N"] - 1))

# Run the iLQR optimization
optimal_trajectory, optimal_controls = controller.run(x_init, u_init)

# Use the optimized trajectory and controls for your system
print("Optimal state trajectory:", optimal_trajectory)
print("Optimal control inputs:", optimal_controls)
```
