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

# whether to plot all timesteps and save the time step data
track = True
H1_sq = 0


x = np.linspace(-0.5, 0.5, 513)
h = x[1] - x[0]


# Set parameters
m = 2
c = np.zeros_like(x)
tau = 0.00625
eps = 1e-3             #1.0**(-3)
M = 0.5
b = (np.sqrt(3) * M / 8)**(2 / 3) 
gamma = 1e-3
h0 = 15
t0 = 1 / gamma * (b / h0)**3
t = t0 * gamma
z = np.maximum(1 / t**(1 / 3) * (b - (1 / (12 * t**(2 / 3))) * x**2), 0)   #z = rho
u = np.zeros_like(z)
delta = 1e-6



class Hist: pass

hist = Hist()

hist.H1_sq = []
hist.phi = []
hist.phiu = []
hist.rho = []
hist.exact = []
hist.Tphi_nu = []
hist.Tpsi_z = []


# JKO scheme
mu = 1 / (delta + m * z**(m-1))
u = gauss(mu, z, tau * gamma, h)

#start = time.process_time()
timestep = np.arange(0, 2, tau)
#stepsize = timestep[1] - timestep[0] = tau
error = 0

for real_t in timestep:
    if real_t == timestep[1]:
        start = time.process_time()
    """
    if real_t == 0:
        plt.ylim([-0.1, 15.1])
        plt.title(r'PME m=2 bbr method t = 0, $\tau = $' + str(tau))
        plt.plot(x, z,label=r'$\rho$')
        #plt.plot(x, phi,label=r'$\phi$')
        ex = np.maximum(1 / t**(1 / 3) * (b - (1 / (12 * t**(2 / 3))) * x**2), 0)
        plt.plot(x, ex, "--",label=r'exact')
        plt.legend(prop={'size': 15})
        plt.savefig(f'{image_root}bbr_rho{real_t:.2}.png', )
        plt.close()
        error = sum(abs((ex - z)[1:] + (ex - z)[:-1]) * h / 2)  #error = (2 / \tau) * \sigma_{n=0}^{2 / \tau} \int |\rho(n*\tau + t0, x) - \rho^(n)(x)| dx
        #print('error = ', error)
        hist.rho.append(z)
        hist.exact.append(ex)
    """


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

"""
# Plot when tau is a multiple of 0.1.
    if (abs((real_t+tau) % 0.1) < 1e-5 or abs(((real_t+tau) % 0.1) - 0.1) < 1e-5):
        plt.ylim([-0.1, 15.1])
        plt.plot(x, z,label=r'$\rho$')
        
        plt.plot(x, ex, "--", label=r'exact')    
        
        print(f'{real_t + tau:.4}: error =  {error}')
        #plt.plot(x, phi,label=r'$\phi$')
        plt.title(r'PME m=2 bbr method t = ' + str(round(real_t+tau, 2)) + r', $\tau = $' + str(tau))
        plt.legend(prop={'size': 15})
        plt.savefig(f'{image_root}bbr_rho{(real_t + tau):.2}.png')
        plt.close()

    if round(real_t+tau, 3) in [0.40, 0.80, 2.00]:
        hist.rho.append(z)
        hist.exact.append(ex)
        print('real_t + stepsize(tau) = ', real_t+tau)
        
    #print(f'Elapsed {(time.process_time() - start):.4}s')
"""

error /= 2 / tau  #error = (2 / \tau) * \sigma_{n=0}^{2 / \tau} \int |\rho(n*\tau + t0, x) - \rho^(n)(x)| dx
error = f"{round(error, 7):.3e}"
realtime = f"{(time.process_time() - start):.3e}"
print('error = ', error)
print(f'Elapsed {realtime}s')
plt.semilogy(hist.H1_sq)
plt.savefig(f'{image_root}_H1_sq.png')
plt.close()

np.save(f'{image_save}/tau={tau}', hist.rho)
np.save(f'{image_save}/exact', hist.exact)

print(f'Plots saved in {image_root}')