import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

'''This file compute the Euler-Maruyama method of tomato system:
We use to solve the SDE system taking the stochastic differential in the parameter 
r_l^tilde dt = r_l dt + sigma_l dB_t
r_i^tilde dt = r_i dt + sigma_i dB_t
gamma^tilde dt=gamma dt +sigma_v dB_t.
'''
########################################################################################################################

num_sims = 5
########################################################################################################################
def brownian_path_sampler(step_size,number_max_of_steps):
    normal_sampler = np.sqrt(step_size)*np.random.randn(number_max_of_steps)
    w_t = np.zeros(number_max_of_steps+1)
    w_t[1:] = np.cumsum(normal_sampler)

    return (normal_sampler,w_t)
########################################################################################################################
T = 70
N = 2 ** 16
dt = T/N

y_init = np.array([0.998999, 0.001, 0.000001, 0.92, 0.08])

beta_p = 0.1
r_l = 0.01
sigma_l = 15.1
sigma_i = 15.1
sigma_v = 15.1
r_i = 0.01
b = 0.075
beta_v = 0.003
gamma = 0.06
theta = 0.4
mu = 0.3
R_0 = np.sqrt(beta_v * mu * b * beta_p / (r_l * r_l * ( r_i + b) * gamma))
########################################################################################################################
def F_func(y, t):
    s_p = y[0]
    l_p = y[1]
    i_p = y[2]
    s_v = y[3]
    i_v = y[4]

    s_p_prime_d = -beta_p * s_p * i_v + r_l * l_p + r_i * i_p
    l_p_prime_d = beta_p * s_p * i_v - b * l_p - r_l * l_p
    i_p_prime_d = b * l_p - r_i * i_p
    s_v_prime_d = - beta_v * s_v * i_p - gamma * s_v + (1 - theta) * mu
    i_v_prime_d = beta_v * s_v * i_p - gamma * i_v + theta * mu
    rhs_d_np_array = np.array([s_p_prime_d, l_p_prime_d, i_p_prime_d, s_v_prime_d, i_v_prime_d])
    return (rhs_d_np_array)

########################################################################################################################
def G_func(y, t):
    s_p = y[0]
    l_p = y[1]
    i_p = y[2]
    s_v = y[3]
    i_v = y[4]

    s_p_prime_s = sigma_l * l_p + sigma_i * i_p
    l_p_prime_s = -sigma_l * l_p
    i_p_prime_s = -sigma_i * i_p
    s_v_prime_s = -sigma_v * s_v
    i_v_prime_s = -sigma_v * i_v
    rhs_s_np_array = np.array([s_p_prime_s, l_p_prime_s, i_p_prime_s, s_v_prime_s, i_v_prime_s])
    return (rhs_s_np_array)

def G_func_prime(y,t):
    s_p = y[0]
    l_p = y[1]
    i_p = y[2]
    s_v = y[3]
    i_v = y[4]

    DG = np.array([[ 0 , sigma_l, sigma_i, 0 , 0],
         [ 0 , -sigma_l, 0, 0, 0],
         [ 0 , 0, -sigma_i, 0 , 0],
         [ 0 , 0, 0, -sigma_v, 0],
         [ 0 , 0, 0, 0, -sigma_v]])

    rhs_s_np_matrix = DG
    return (rhs_s_np_matrix)
########################################################################################################################

dB,B_t = brownian_path_sampler(dt,N) #generating the path

ts = np.arange(0, T, dt)
ys = np.zeros((5,N))

ys[:,0] = y_init

for _ in range(num_sims):

    for i in range(1, ts.size):
        t = (i-1) * dt
        y = ys[:,i-1]
        ys[:,i] = y + F_func(y, t) * dt + G_func(y, t) * dB[i-1] + 0.5 * (G_func(y,t) @ G_func_prime(y,t)) * (dB[i-1] ** 2 - dt)
    #plt.plot(ts, ys[2,:],color = 'r', label="Stochastic Solution")
########################################################################################################################
######################################### DETERMINISTIC SOLUTION #######################################################
def rhs(y, t_zero):
    s_p = y[0]
    l_p = y[1]
    i_p = y[2]
    s_v = y[3]
    i_v = y[4]

    s_p_prime = r_l * l_p + r_i * i_p - beta_p * s_p * i_v
    l_p_prime = beta_p * s_p * i_v - b * l_p - r_l * l_p
    i_p_prime = b * l_p - r_i * i_p
    s_v_prime = - beta_v * s_v * i_p - gamma * s_v + (1 - theta) * mu
    i_v_prime = beta_v * s_v * i_p - gamma * i_v + theta * mu
    rhs_np_array = np.array([s_p_prime, l_p_prime, i_p_prime, s_v_prime, i_v_prime])
    return (rhs_np_array)
y_zero = np.array([0.998999, 0.001, 0.000001, 0.92, 0.08])
t = np.linspace(0, T, N)
sol = odeint(rhs, y_zero, t)
########################################################################################################################
########################################################################################################################
fig, axs = plt.subplots(3, 2, sharex=True)
fig.subplots_adjust(left=0.08, right=0.98, wspace=0.3)
ax0 = axs[0, 0]
ax0.plot(ts, ys[0,:],color = 'r', label="Stochastic Solution")
ax0.plot(t, sol[:, 0], color ='b',label="Deterministic Solution")
ax0.set_xlabel('$t$')
ax0.set_ylabel('$S_p$')
ax0.grid(True)

ax1 = axs[0, 1]
ax1.plot(ts, ys[1,:],color = 'r', label="Stochastic Solution")
ax1.plot(t, sol[:, 1], color ='b',label="Deterministic Solution")
ax1.set_xlabel('$t$')
ax1.set_ylabel('$L_p$')
ax1.grid(True)

ax2 = axs[1, 0]
ax2.plot(ts, ys[2,:],color = 'r', label="Stochastic Solution")
ax2.plot(t, sol[:, 2], color ='b',label="Deterministic Solution")
ax2.set_xlabel('$t$')
ax2.set_ylabel('$I_p$')
ax2.grid(True)

ax3 = axs[1, 1]
ax3.plot(ts, ys[3,:],color = 'r', label="Stochastic Solution")
ax3.plot(t, sol[:, 3], color ='b',label="Deterministic Solution")
ax3.set_xlabel('$t$')
ax3.set_ylabel('$S_v$')
ax3.grid(True)

ax4 = axs[2, 0]
ax4.plot(ts, ys[4,:],color = 'r', label="Stochastic Solution")
ax4.plot(t, sol[:, 4], color ='b',label="Deterministic Solution")
ax4.set_xlabel('$t$')
ax4.set_ylabel('$I_v$')
ax4.grid(True)

plt.tight_layout()

ax5 = axs[2, 1]
ax5.plot(ts, B_t[0:N], color='k', label="Stochastic Solution")
#ax5.plot(ts, b_t[0:N], color='b', label="Stochastic Solution")
ax5.set_xlabel('$t$')
ax5.set_ylabel('$B_t$')
ax5.grid(True)

plt.tight_layout()

print(R_0)
########################################################################################################################
########################################################################################################################
plt.show()
