def RK4(dynamics, x, u, h = 0.001):
    '''
    Fourth order Runge-Kutta explicit integration

    :param dynamics: function that returns x_dot as a function of state and input
    :param x: state
    :oaram u: action
    :return : state at time t + h
    '''
    f1 = dynamics(x, u)
    f2 = dynamics(x + 0.5*h*f1, u)
    f3 = dynamics(x + 0.5*h*f2, u)
    f4 = dynamics(x + 1.0*h*f3, u)
    return x + (h/ 6.0)*(f1 + 2*f2 + 2*f3 + f4)

