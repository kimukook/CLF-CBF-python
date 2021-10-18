import numpy as np


class OdeSolver:
    """

    """
    def __init__(self, system, dt):
        self.f = system.f
        self.g = system.g
        self.dt = dt

    def dynamic(self, x, u):
        return self.f(x) + self.g(x) @ np.atleast_2d(u)

    def time_marching(self, x, u):
        return self.runge_kutta4(x, u)

    def runge_kutta4(self, x, u):
        # issue might be raised on f1->f4, all has to be 1D row-vector in numpy
        f1 = self.dynamic(x, u).T[0]
        f2 = self.dynamic(x + self.dt/2 * f1, u).T[0]
        f3 = self.dynamic(x + self.dt/2 * f2, u).T[0]
        f4 = self.dynamic(x + self.dt * f3, u).T[0]
        x_new = x + self.dt/6 * (f1 + 2*f2 + 2*f3 + f4)
        return x_new
