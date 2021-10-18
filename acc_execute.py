import numpy as np
import acc
import define_system
import cbf_clf_qp as ccq
import ode_solver
import matplotlib.pyplot as plt

parameters = {
    'f0': .1,
    'f1': 5,
    'f2': .25,
    'v0': 10,
    'm': 1650,
    'T': 1.8,
    'cd': .3,
    'vd': 24,
    'G': 9.8,
    'udim': 1
}

vehicle = acc.AdaptiveCruiseControl(parameters)
ds = define_system.ControlAffineSystem(vehicle)

QPoption = ccq.CbfClfQpOptions()
QPoption.set_option('u_max', np.array([.3 * parameters['G'] * parameters['m']]))
QPoption.set_option('u_min', np.array([-.3 * parameters['G'] * parameters['m']]))
QPoption.set_option('clf_lambda', 5.0)
QPoption.set_option('cbf_gamma', 5.0)
QPoption.set_option('weight_input', np.array([2/parameters['m']**2]))
QPoption.set_option('weight_slack', 2e-2)

qp = ccq.CbfClfQp(ds, QPoption)


T = 20
dt = .02
x0 = np.array([[0, 20, 100]]).T
time_steps = int(np.ceil(T / dt))

ode_sol = ode_solver.OdeSolver(ds, dt)

# initialize the input matrices
xt = np.zeros((3, time_steps))
tt = np.zeros((1, time_steps))
ut = np.zeros((1, time_steps))

slackt = np.zeros((1, time_steps))
Vt = np.zeros((1, time_steps))
Bt = np.zeros((1, time_steps))
xt[:, 0] = np.copy(x0.T[0])

for t in range(time_steps-1):
    if t % 100 == 0:
        print(f't = {t}')
    # print(t)

    # reference control input u_ref
    Fr = vehicle.getFr(xt[:, t])

    # solve for control u at current time step
    u, delta, B, V, feas = qp.cbf_clf_qp(xt[:, t], Fr)
    if feas == -1:
        # infeasible
        break
    else:
        pass

    ut[:, t] = np.copy(u)
    slackt[:, t] = np.copy(delta)
    Vt[:, t] = np.copy(V)
    Bt[:, t] = np.copy(B)

    # propagate the system with control u using RK4
    xt[:, t + 1] = ode_sol.time_marching(xt[:, t], u)


def state_velocity(xt, dt, T):
    t = np.arange(0, T, dt)
    fig = plt.figure(figsize=[16, 9])
    plt.grid()
    plt.plot(t, xt[1, :], linewidth=3, color='magenta')
    plt.plot(t, 24 * np.ones(t.shape[0]), 'k--')
    plt.title('State - Velocity')
    plt.ylabel('v')
    # plt.show()
    plt.savefig('velocity.png', format='png', dpi=300)
    plt.close(fig)


def state_relative_distance(xt, dt, T):
    t = np.arange(0, T, dt)
    fig = plt.figure(figsize=[16, 9])
    plt.grid()
    plt.plot(t, xt[2, :], linewidth=3, color='black')
    plt.ylim(0, 100)
    plt.title('State - Relative distance')
    plt.ylabel('z')
    # plt.show()
    plt.savefig('relative_distance.png', format='png', dpi=300)
    plt.close(fig)

    
def slack(slack, dt, T):
    t = np.arange(0, T, dt)
    fig = plt.figure(figsize=[16, 9])
    plt.grid()
    plt.plot(t[:-1], slack[0, :-1], linewidth=3, color='orange')
    plt.title('Slack')
    plt.ylabel('B(x)')
    # plt.show()
    plt.savefig('slack.png', format='png', dpi=300)
    plt.close(fig)


def cbf(Bt, dt, T):
    t = np.arange(0, T, dt)
    fig = plt.figure(figsize=[16, 9])
    plt.grid()
    plt.plot(t[:-1], Bt[0, :-1], linewidth=3, color='red')
    plt.title('cbf')
    plt.ylabel('B(x)')
    # plt.show()
    plt.savefig('cbf.png', format='png', dpi=300)
    plt.close(fig)


def clf(Vt, dt, T):
    t = np.arange(0, T, dt)
    fig = plt.figure(figsize=[16, 9])
    plt.grid()
    plt.plot(t[:-1], Vt[0, :-1], linewidth=3, color='cyan')
    plt.title('clf')
    plt.ylabel('V(x)')
    # plt.show()
    plt.savefig('clf.png', format='png', dpi=300)
    plt.close(fig)
    
    
def control(u, dt, T):
    u_max = .3 * 9.8 * 1650
    t = np.arange(0, T, dt)
    fig = plt.figure(figsize=[16, 9])
    plt.grid()
    plt.plot(t[:-1], u[0, :-1], linewidth=3, color='dodgerblue')
    plt.plot(t, u_max * np.ones(t.shape[0]), 'k--')
    plt.plot(t, -u_max * np.ones(t.shape[0]), 'k--')
    plt.title('control')
    plt.ylabel('u(t, x)')
    # plt.show()
    plt.savefig('control.png', format='png', dpi=300)
    plt.close(fig)


state_velocity(xt, dt, T)
state_relative_distance(xt, dt, T)
cbf(Bt, dt, T)
clf(Vt, dt, T)
slack(slackt, dt, T)
control(ut, dt, T)



