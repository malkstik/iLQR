def RK4(dynamics, x, u, dt = 0.001):
    '''
    Fourth order Runge-Kutta explicit integration

    :param dynamics: function that returns x_dot as a function of state and input
    :param x: state
    :oaram u: action
    :return : state at time t + h
    '''
    f1 = dynamics(x, u) * dt
    f2 = dynamics(x + 0.5*f1, u) * dt
    f3 = dynamics(x + 0.5*f2, u) * dt
    f4 = dynamics(x + 1.0*f3, u) * dt
    return x + (f1 + 2*f2 + 2*f3 + f4)/6.0