import numpy as np
import matplotlib.pyplot as plt
import os
import time
from numba import njit

image_root = "../images/bbr/"
os.makedirs(image_root, exist_ok = True)
image_save = "../images/bbr_tau/"
os.makedirs(image_save, exist_ok = True)

# Wasserstein distance \int \phi d\nu + \int \phi^c d\mu
def w2(phi, psi, mu, nu):
    return np.sum(phi * nu + psi * mu)

@njit
def gauss(mu, z, tau, h):
    a = np.full_like(mu, - tau / h**2)
    b =  mu + 2 * tau / h**2
    c = np.full_like(mu, - tau / h**2)
    u = mu * z**m
    c[1] = c[1] / b[1]
    u[1] = u[1] / b[1]
    for i in range(2,len(x)-1):
        c[i] = c[i] / (b[i] - a[i] * c[i-1])
        u[i] = (u[i] - a[i] * u[i-1]) / (b[i] - a[i] * c[i-1])
    for i in range(len(x)-3, 0, -1):
    #for i in reversed(range(1, len(x)-2)):
        u[i] = u[i] - c[i] * u[i+1]
    
    return u


# ascent step of J(phi) = \int phi dnu + \int phi^c dmu
# fills phi and phi_c, returns new sigma
# using Jacobian push fofward (push_forward2) 
# common ascent scheme
def bbr(tau): # JKO scheme
# Initialisation of parameters
    t = t0 * gamma
    z = np.maximum(1 / t**(1 / 3) * (b - (1 / (12 * t**(2 / 3))) * x**2), 0)   #z = rho
    mu = 1 / (delta + m * z**(m-1))
    u = gauss(mu, z, tau * gamma, h)
    timestep = np.arange(0, 2, tau)
    #stepsize = timestep[1] - timestep[0] = tau
    
    error = 0
    start = time.process_time()
    # JKO scheme
    for real_t in timestep:
        u[0] = u[-1] = 0
        mu = 1 / (delta + m * z**(m-1))
        #print(mu)
        #mu * nu[1:-1] - tau / h**2 * (nu[:-2] -2*nu[1:-1] + nu[2:]) = mu * z[1:-1]**m
        u = gauss(mu, z, tau * gamma, h)
        #print(np.max(np.abs((mu[1:-1] + 2*tau/h**2) * u[1:-1] - tau / h**2 * (u[:-2] + u[2:]) - mu[1:-1] * z[1:-1]**m)))
        z = np.maximum(z + mu * (u - z**m), 0)
        t = (real_t + tau + t0) * gamma
        ex = np.maximum(1 / t**(1 / 3) * (b - (1 / (12 * t**(2 / 3))) * x**2), 0)
        area = sum((ex[1:] + ex[:-1]) * h / 2)   #trapezoidal formula 
        error += sum(abs((ex - z)[1:] + (ex - z)[:-1]) * h / 2)  #error = (2 / \tau) * \sigma_{n=0}^{2 / \tau} \int |\rho(n*\tau + t0, x) - \rho^(n)(x)| dx

    error /= 2 / tau  #error = (2 / \tau) * \sigma_{n=0}^{2 / \tau} \int |\rho(n*\tau + t0, x) - \rho^(n)(x)| dx
    error = f'{round(error, 7):.2e}'
    realtime = f'{(time.process_time() - start):.3g}'
    
    return error, realtime, area



# whether to plot all timesteps and save the time step data
track = True
H1_sq = 0

# Set parameters
x = np.linspace(-0.5, 0.5, 513)
h = x[1] - x[0]
m = 2
c = np.zeros_like(x)
tau = 0.4
eps = 1e-3             #1.0**(-3)
M = 0.5
b = (np.sqrt(3) * M / 8) ** (2 / 3)
gamma = 1e-3
h0 = 15
t0 = 1 / gamma * (b / h0) ** 3
delta = 1e-6


# Load the respective function first.
print('tau = ', tau)
start = time.process_time()
error, realtime, area = bbr(tau)
print('error = ', error)
print(f'Elapsed {realtime}s')
print('')


# Save the ERROR and TIME in a text file.
with open('Codes/result/result_bbr512.tex', 'w') as f:
    f.write('\\begin{tabular}{llll} \n')
    f.write('\hline \n')
    f.write('$\\tau$  & $N_\\tau$  &  Error & Times$(s)$  \\\ \n')
    f.write('\hline \hline \n')
    
    
    # Repeat the operation of halving the value of tau 7 times.
    for i in range(8):
        print('tau = ', tau)
        error, realtime, area = bbr(tau)
        print('error = ', error)
        print(f'Elapsed {realtime}s')
        print('')
        #String to be saved in a text file.
        f.write(f'{tau}  & {int(2 / tau)} & \\num{{{error}}} & \\num{{{realtime}}} \\\ \n') 

        tau /= 2
        if i == 6:
            tau = 1e-4
    f.write('\hline \n')
    f.write('\end{tabular} \n')
