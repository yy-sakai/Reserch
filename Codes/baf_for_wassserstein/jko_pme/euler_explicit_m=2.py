import numpy as np
import matplotlib.pyplot as plt
import os
import time

image_root = "../images/euler/"
os.makedirs(image_root, exist_ok = True)
image_save = "../images/euler_tau/"
os.makedirs(image_save, exist_ok = True)
# Wasserstein distance \int \phi d\nu + \int \phi^c d\mu
def w2(phi, psi, mu, nu):
    return np.sum(phi * nu + psi * mu)

# ascent step of J(phi) = \int phi dnu + \int phi^c dmu
# fills phi and phi_c, returns new sigma
# using Jacobian push fofward (push_forward2) 
# common ascent scheme

# whether to plot all timesteps and save the time step data
track = True
H1_sq = 0


x = np.linspace(-0.5, 0.5, 101)
h = x[1] - x[0]


# Set parameters
m = 2
c = np.zeros_like(x)
tau = 0.001
eps = 1e-3             #1.0**(-3)
M = 0.5
b = (np.sqrt(3) * M / 8)**(2 / 3) 
gamma = 1e-3
h0 = 15
t0 = 1 / gamma * (b / h0)**3
t = t0 * gamma
nu = np.maximum(1 / t**(1 / 3) * (b - (1 / (12 * t**(2 / 3))) * x**2), 0)   #nu = rho


class Hist: pass

hist = Hist()

hist.H1_sq = []
hist.phi = []
hist.phiu = []
hist.rho = []
hist.exact = []
hist.Tphi_mu = []
hist.Tpsi_nu = []


# JKO scheme

start = time.process_time()
timestep = np.arange(0, 2, tau)
#stepsize = timestep[1] - timestep[0] = tau
error = 0
count = 0
for real_t in timestep:
    if real_t == 0:
        plt.ylim([-0.1, 15.1])
        plt.title(r'back-and-forth update $\mu$ and $\nu$. Example 1:  Iterate ' + str(count))
        plt.plot(x, nu,label=r'$\nu$')
        #plt.plot(x, phi,label=r'$\phi$')
        u = np.maximum(1 / t**(1 / 3) * (b - (1 / (12 * t**(2 / 3))) * x**2), 0)
        plt.plot(x, u, "--",label=r'exact')
        plt.legend(prop={'size': 15})
        plt.savefig(f'{image_root}Tphi_mu,Tpsi_nu{real_t:.2}.png', )
        plt.close()
        error = sum(abs((u - nu)[1:] + (u - nu)[:-1]) * h / 2)  #error = (2 / \tau) * \sigma_{n=0}^{2 / \tau} \int |\rho(n*\tau + t0, x) - \rho^(n)(x)| dx
        #print('error = ', error)
        hist.rho.append(nu)
        hist.exact.append(u)

    nu[0] = nu[-1] = 0
    nu[1:-1] = nu[1:-1] + ((tau * gamma)/ h**2) * (nu[:-2]**m -2*nu[1:-1]**m + nu[2:]**m)

    if count % 10 == 9:
        plt.ylim([-0.1, 15.1])
        plt.plot(x, nu,label=r'$\nu$')
        
        t = (real_t + tau + t0) * gamma
        u = np.maximum(1 / t**(1 / 3) * (b - (1 / (12 * t**(2 / 3))) * x**2), 0)
        area = sum((u[1:] + u[:-1]) * h / 2)   #trapezoidal formula
        print(area)
        plt.plot(x, u, "--", label=r'exact')    
        
        #plt.plot(x, phi,label=r'$\phi$')
        plt.legend(prop={'size': 15})
        plt.savefig(f'{image_root}Tphi_mu,Tpsi_nu{(real_t + tau):.4}.png', )
        plt.close()
    
    error += sum(abs((u - nu)[1:] + (u - nu)[:-1]) * h / 2)  #error = (2 / \tau) * \sigma_{n=0}^{2 / \tau} \int |\rho(n*\tau + t0, x) - \rho^(n)(x)| dx
    print('error = ', error)

    if round(real_t+tau, 3) in [0.40, 0.80, 2.00]:
        hist.rho.append(nu)
        hist.exact.append(u)
        print('real_t + stepsize(tau) = ', real_t+tau)
        
    print(f'{real_t + tau:.4}:')
    print(f'Elapsed {(time.process_time() - start):.4}s')
    
    count += 1

error /= 2 / tau  #error = (2 / \tau) * \sigma_{n=0}^{2 / \tau} \int |\rho(n*\tau + t0, x) - \rho^(n)(x)| dx
error = round(error, 4)
print('error = ', error)
plt.semilogy(hist.H1_sq)
plt.savefig(f'{image_root}_H1_sq.png')
plt.close()

np.save(f'{image_save}/tau="{tau}', hist.rho)
np.save(f'{image_save}/exact', hist.exact)

print(f'Plots saved in {image_root}')