'''
=====================================
Author  :  Muhan Zhao
Date    :  Feb. 16, 2020
Location:  UC San Diego, La Jolla, CA
=====================================
'''

import numpy as np
import cvxpy as cp


class OptionsClass:
    """
    Options Class
    """

    def __init__(self):
        self.options = None
        self.solverName = 'None'

    def set_option(self, key, value):
        try:
            if type(value) is self.options[key][2]:
                self.options[key][0] = value
            else:
                print(f"The type of value for the keyword '{key}' should be '{self.options[key][2]}'.")
        except:
            raise ValueError('Incorrect option keyword or type: ' + key)

    def get_option(self, key):
        try:
            value = self.options[key][0]
            return value
        except:
            raise ValueError('Incorrect option keyword: ' + key)

    def reset_options(self, key):
        try:
            self.options[key] = self.options[key][1]
        except:
            raise ValueError('Incorrect option keyword: ' + key)


class CbfClfQpOptions(OptionsClass):
    def __init__(self):
        OptionsClass.__init__(self)
        self.setup()
        self.solver_name = 'CBF-CLF'

    def setup(self):
        self.options = {
            # [Current value, default value, type]
            'u_max': [None, None, np.ndarray],
            'u_min': [None, None, np.ndarray],
            'clf_lambda': [None, 5, float],
            'cbf_gamma': [None, 5, float],
            'weight_input': [None, None, np.ndarray],
            'weight_slack': [None, 2e-2, float],
        }

    # def define_slack(self):
        # TODO


class CbfClfQp:
    """
    This is the implementation of the vanilla CBF-CLF-QP method. The optimization problem is:

            min (u-u_ref).T * H * (u-u_ref) + p * delta**2
            s.t. L_f V(x) + L_g V(x) * u + lambda * V(x) <= delta  ---> CLF constraint
                 L_f B(x) + L_g B(x) * u + gamma * B(x) >= 0  ---> CBF constraint

    Input:
    :param  system  :   The dynamic system of interest, containing CBF, CLF, and their Lie derivatives
    :param  x       :   The current state x
    :param  u_ref   :   The reference control input
    :param  slack   :   The slack activated or not, 1 -> activate while 0 -> not activate
    :param  verbose :   Show the optimization log or not
    """
    def __init__(self, system, option_class):
        if hasattr(system, 'udim'):
            self.udim = system.udim
        else:
            raise KeyError('udim is not given in the system dynamic')

        self.cbf = system.cbf

        # todo check lf.lg/cbf clfs symbolic expression and their size!
        self.lf_cbf = system.lf_cbf
        self.lg_cbf = system.lg_cbf

        self.clf = system.clf
        self.lf_clf = system.lf_clf
        self.lg_clf = system.lg_clf

        # todo take input from the option class
        self.weight_input = np.atleast_2d(option_class.get_option('weight_input'))
        self.weight_slack = np.atleast_2d(option_class.get_option('weight_slack'))
        self.H = None
        self.slack_H = None

        # todo
        self.A = None
        self.b = None

        # Hyperparameters: CLF <- Lambda & CBF <- Gamma
        self.clf_lambda = option_class.get_option('clf_lambda')
        self.cbf_gamma = option_class.get_option('cbf_gamma')

        self.u_max = option_class.get_option('u_max')
        if self.u_max.shape != (self.udim,):
            raise ValueError('The size of u_max should be udim-by-, a one dimensional vector in python.')
        self.u_min = option_class.get_option('u_min')
        if self.u_min.shape != (self.udim,):
            raise ValueError('The size of u_min should be udim-by-, a one dimensional vector in python.')

        self.with_slack = None

    def cbf_clf_qp(self, x, u_ref, with_slack=1, verbose=0):
        """

        :param x         :   The current state
        :param u_ref     :   A real number of 1D vector with shape (udim,)
        :param with_slack:   Indicator if there is slack variable
        :param verbose   :   Indicator if QP info is displayed
        :return:
        """
        inf = np.inf
        self.with_slack = with_slack

        slack = None
        if u_ref is None:
            u_ref = np.zeros(self.udim)
        else:
            if u_ref.shape != (self.udim,):
                raise ValueError(f'u_ref should have the shape size (u_dim,), now it is {u_ref.shape}')

        # Read the weight input and build up the matrix H in the cost function
        if self.weight_input.shape == (1, 1):
            # Weight input is a scalar
            self.H = self.weight_input * np.eye(self.udim)

        elif self.weight_input.shape == (self.udim, 1):
            # Weight_input is a vector, use it to form the diagonal of the H matrix
            self.H = np.diag(self.weight_input)

        elif self.weight_input.shape == (self.udim, self.udim):
            # Weight_input is a udim * udim matrix
            self.H = np.copy(self.weight_input)
        else:
            self.H = np.eye(self.udim)

        V = self.clf(x)
        lf_V = self.lf_clf(x)
        lg_V = self.lg_clf(x)

        B = self.cbf(x)
        lf_B = self.lf_cbf(x)
        lg_B = self.lg_cbf(x)

        if self.with_slack:
            # slack variable is activated
            # Constraints: A [u; slack] <= b
            # LfV + LgV * u + lambda * V <= slack
            # LfB + LgB * u + gamma * B  >= 0
            lg_V = np.hstack((lg_V, -np.ones((1, 1))))
            lg_B = np.hstack((-lg_B, np.zeros((1, 1))))

            self.A = np.vstack((lg_V, lg_B))
            self.b = np.hstack((-lf_V - self.clf_lambda * V, lf_B + self.cbf_gamma * B))

            # make sure that b is just a 1D vector with the shape (udim+1,)
            self.b = np.atleast_2d(self.b)[0]

            # Slack -> unconstrained
            u_min = np.hstack((self.u_min, -inf * np.ones(1)))
            u_max = np.hstack((self.u_max, inf * np.ones(1)))

            u = cp.Variable(self.udim + 1)

            # H_new = [H, 0; 0, p]
            self.slack_H = np.hstack((self.H, np.zeros((1, 1))))
            self.slack_H = np.vstack((self.slack_H, np.hstack((np.zeros((1, 1)), self.weight_slack * np.ones((1, 1))))))

            # Cost -> (u-u_ref)' * H_new * (u-u_ref) + p * delta**2
            #      -> (1/2) * [u slack]' * H_new * [u slack] - [u slack]' * H_new * [u_ref 0]
            u_ref = np.hstack((u_ref, np.zeros(1)))
            objective = cp.Minimize((1/2) * cp.quad_form(u, self.slack_H) - (self.slack_H @ u_ref).T @ u)

            # Constraints: A * u <= b and u_min, u_max
            constraints = [u_min <= u, u <= u_max, self.A @ u <= self.b]
            # constraints = [self.u_min <= u, u <= self.u_max, np.eye(2) @ u <= np.zeros(2)]

            problem = cp.Problem(objective, constraints)

            problem.solve()

            # what if infeasible?
            if problem.status != 'infeasible':
                slack = u.value[-1]
                u = u.value[:self.udim]
                feas = 1
            else:
                u = None
                slack = None
                feas = -1

        else:
            # Slack variable is not activated:
            # Constraints: A u <= b
            # LfV + LgV * u + lambda * V <= 0
            # LfB + LgB * u + gamma * B >= 0
            self.A = np.vstack((lg_V, -lg_B))
            # b -> one dimensional vector
            self.b = np.hstack((-lf_V - self.clf_lambda * V, lf_B + self.cbf_gamma * B))
            self.b = np.atleast_2d(self.b)[0]

            u = cp.Variable(self.udim)

            # Cost -> (u-u_ref)' * H * (u-u_ref) -> (1/2) * u'*H*u - u'*H*u_ref
            objective = cp.Minimize((1/2)*cp.quad_form(u, self.H) - (self.H @ u_ref).T @ u)

            # cons: A * u <= b and u_min, u_max
            constraints = [self.u_min <= u, u <= self.u_max, self.A @ x <= self.b]

            problem = cp.Problem(objective, constraints)

            problem.solve()

            if problem.status != 'infeasible':
                u = u.value
                feas = 1
            else:
                u = None
                feas = -1

        return u, slack, B, V, feas



