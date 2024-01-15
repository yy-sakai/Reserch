import numpy as np
import matplotlib.pyplot as plt
import os
import time
from premise_of_baf.c_transform import c_transform
from premise_of_baf.push_forward_jacobian import push_forward2
from premise_of_baf.push_forward_jacobian import lap_solve_modified
from numba import njit

image_root = "../images/baf_m=2/"
os.makedirs(image_root, exist_ok = True)

image_save = "../images/baf_tau/"
os.makedirs(image_save, exist_ok = True)

# Wasserstein distance \int \phi d\nu + \int \phi^c d\mu
def w2(phi, psi, mu, nu):
    return np.sum(phi * nu + psi * mu)

@njit
def gauss(f, theta_1, theta_2):
    a = np.full_like(x, - theta_2 / h**2)
    b = np.full_like(x, theta_1 + 2 * theta_2 / h**2)
    c = np.full_like(x, - theta_2 / h**2)
    c[0] = c[0] / b[0]
    f[0] = f[0] / b[0]
    for i in range(1,len(x)):
        c[i] = c[i] / (b[i] - a[i] * c[i-1])
        f[i] = (f[i] - a[i] * f[i-1]) / (b[i] - a[i] * c[i-1])
    for i in range(len(x)-3, 0, -1):
    #for i in reversed(range(1, len(x)-2)):
        f[i] = f[i] - c[i] * f[i+1]
    
    return f

# ascent step of J(phi) = \int phi dnu + \int phi^c dmu
# fills phi and phi_c, returns new sigma
# using Jacobian push fofward (push_forward2) 
# common ascent scheme

def ascent(phi, phi_c, mu, nu):
    phi_c, _ = c_transform(x, tau * phi, x)                        # 1-1  phi_c, _ = c_transform(x, phi, p)
    phi_c /= tau

    nmu = np.max(np.abs(mu))
    theta_1 = 1 / (2 * gamma)
    theta_2 = tau * nmu
    
    pfwd = push_forward2(mu, tau * phi, h)           # 1-2-1     pfwd : T_{\phi\#}\mu = \mu(x - \tau \nabla \phi(x))|det(I - \tau D^2\phi_c))|
    rho = nu - pfwd                                 # 1-2-2     rho = \nu - T_{\phi\#}\mu　＝ \delta U^*(- \phi) - T_{\phi\#}\mu
    #TODO: This is by far the slowest part of the algorithm
    
    # In one dimension, Gaussian elimination is faster than the Fast Fourier Transform.
    #lp = lap_solve_modified(rho, theta_1, theta_2)                             # 1-2-3     lp: \nabla_{\dot{H}^1} J(\phi_n) = (- \Delta)^{-1} * rho
    lp = gauss(rho, theta_1, theta_2)
    
    phi += lp                               # 1-2-4   phi_{n + 1/2} = phi_n + sigma * lp
#####################################################################                     
    phi_c, _ = c_transform(x, tau * phi, x)                    # 2    psi_{n + 1/2} = (phi_{n + 1/2})^c
    phi_c /= tau
    H1_sq = np.mean(rho * lp)                        #######       ?
    return H1_sq, phi, phi_c, pfwd   #  ?

# whether to plot all timesteps and save the time step data
track = True
H1_sq = 0

# Set parameters
x = np.linspace(-0.5, 0.5, 4001)
h = x[1] - x[0]
m = 2
c = np.zeros_like(x)
tau = 0.0001
eps = 1e-5             #1.0**(-3)
M = 0.5
b = (np.sqrt(3) * M / 8)**(2 / 3) 
gamma = 1e-3
h0 = 15
t0 = 1 / gamma * (b / h0)**3
t = t0 * gamma
nu = np.maximum(1 / t**(1 / 3) * (b - (1 / (12 * t**(2 / 3))) * x**2), 0)   #nu = rho
mu = nu

# U(\rho) = 1 / m-1 \int \rho^m dx 
phi = - gamma * (m / (m - 1)) * nu ** (m - 1)  #\phi_0 = \phi^(0) = -\delta U(\nu^(0)) =  \delta U(\rho^(0))
psi = np.zeros_like(nu)  


class Hist: pass

hist = Hist()

hist.H1_sq = []
hist.phi = []
hist.phiu = []
hist.rho = []
hist.exact = []
hist.Tphi_mu = []
hist.Tpsi_nu = []
hist.save_error = []


# JKO scheme

H1_sq, phi, psi, pfwd  = ascent(phi, psi, mu, nu)
psi_c, _ = c_transform(x, tau * psi, x)


timestep = np.arange(0, 2, tau)
#stepsize = timestep[1] - timestep[0] = tau
print('tau = ', tau)
print('start')
error = 0
start = time.process_time()

for real_t in timestep:
    #if real_t == timestep[1]:
     #   start = time.process_time()
    diff = 1
    count = 0
# The back-and-forth scheme for solving J(phi) and I(psi)
    while diff >= eps:
        if count > 100:
            break
        
        if real_t == 0 and count == 0:
            plt.ylim([-0.1, 15.1])
            plt.title(r'PME m=2 baf method t = 0, $\tau = $' + str(tau))
            plt.plot(x, nu,label=r'$\rho$')
            #plt.plot(x, phi,label=r'$\phi$')
            u = np.maximum(1 / t**(1 / 3) * (b - (1 / (12 * t**(2 / 3))) * x**2), 0)
            plt.plot(x, u, "--",label=r'exact')
            plt.legend(prop={'size': 15})
            plt.savefig(f'{image_root}baf_rho{real_t:.2}.png')
            plt.close()
            
            plt.plot(x, phi,label=r'$\phi$')
            plt.legend()
            plt.savefig(f'{image_root}phi{count:04}.png', )
            plt.close()
            error += (tau / 2) * sum(abs((u - nu)[1:] + (u - nu)[:-1]) * h / 2)  #error = (2 / \tau) * \sigma_{n=0}^{2 / \tau} \int |\rho(n*\tau + t0, x) - \rho^(n)(x)| dx
            #print('error = ', error)
            hist.rho.append(nu)
            hist.exact.append(u)
        
            
            
        nu = (((m - 1) / (m * gamma)) * np.maximum(c - phi, 0)) ** (1 / (m - 1)) # \rho_*(x) = \delta U^*(- \phi) 
        H1_sq, phi, psi, pfwd  = ascent(phi, psi, mu, nu)  # phi = phi_{k + 1/2}, psi = psi_{k + 1/2}
        
        
        # Calculate residual $||\nabla U^*(- \varphi) - T_{\phi \#} \mu||_{L^1(\Omega)}
        #L1 norm
        diff = sum(abs(nu - pfwd) * h)
        
        #print(f'{real_t + tau:.4}:(H¹)² = {H1_sq:.3}, diff = {diff:.5}, baf_roop = {count}')
            
        hist.H1_sq.append(H1_sq)
        #hist.phi.append(np.float32(phi))
        hist.Tphi_mu.append(np.float32(pfwd))
        
        ##################################################################################
        
        psi_c, _ = c_transform(x, tau * psi, x)        # 3-1  phi_c, _ = c_transform(x, phi, p)
        psi_c /= tau
        
        nu = (((m - 1)/ (m * gamma)) * np.maximum(c - psi_c, 0)) ** (1 / (m - 1)) # nu = T_{\psi \#} \delta U^* (- \psi^c)
        H1_sq, psi, phi, pfwd = ascent(psi, phi, nu, mu)
        
        #print(f'{real_t + tau:.4}:(H¹)² = {H1_sq:.3}, diff = {diff:.5}, baf_roop = {count}')
        
        hist.H1_sq.append(H1_sq)
        hist.Tpsi_nu.append(np.float32(pfwd))
        hist.phi.append(np.float32(phi))
        
        count+= 1

    mu = (((m - 1) / (m * gamma)) * np.maximum(c - phi, 0)) ** (1 / (m - 1)) # 


    t = (real_t + tau + t0) * gamma
    u = np.maximum(1 / t**(1 / 3) * (b - (1 / (12 * t**(2 / 3))) * x**2), 0)
    area = sum((u[1:] + u[:-1]) * h / 2)   #trapezoidal formula 
    error += (tau / 2) * sum(abs((u - nu)[1:] + (u - nu)[:-1]) * h / 2)  #error = (2 / \tau) * \sigma_{n=0}^{2 / \tau} \int |\rho(n*\tau + t0, x) - \rho^(n)(x)| dx

#   Plot when tau is a multiple of 0.1.
    
    if (abs((real_t+tau) % 0.1) < 1e-5 or abs(((real_t+tau) % 0.1) - 0.1) < 1e-5):
        plt.ylim([-0.1, 15.1])
        plt.plot(x, nu,label=r'$\rho$')        
        plt.plot(x, u, "--", label=r'exact')  
        
        print(f'{real_t + tau:.4}: error =  {error}')
        #plt.plot(x, phi,label=r'$\phi$')
        plt.title(r'PME m=2 baf method t = ' + str(round(real_t+tau, 2)) + r', $\tau = $' + str(tau))
        plt.legend(prop={'size': 15})
        plt.savefig(f'{image_root}baf_rho{(real_t + tau):.2}.png')
        plt.close()
        
    

    if round(real_t+tau, 5) in [0.40, 0.80, 2.00]:
        hist.rho.append(nu)
        hist.exact.append(u)
        print('real_t + stepsize(tau) = ', real_t+tau)
        
    #print(f'Elapsed {(time.process_time() - start):.4}s')

#error /= 2 / tau  #error = (2 / \tau) * \sigma_{n=0}^{2 / \tau} \int |\rho(n*\tau + t0, x) - \rho^(n)(x)| dx
error = f"{round(error, 7):.2e}"
realtime = f'{(time.process_time() - start):.3g}'
print('error = ', error)
print(f"Elapsed {realtime}s")
plt.semilogy(hist.H1_sq)
plt.savefig(f'{image_root}_H1_sq.png')
plt.close()

np.save(f'{image_save}/tau={tau}', hist.rho)
np.save(f'{image_save}/exact', hist.exact)

# Save error for each x at t=2.0
save_error = abs((u - nu)[1:] + (u - nu)[:-1]) * h / 2  
hist.save_error.append(save_error)
#print(save_error)

#plot error graph
center_x = np.array((x[1:] + x[:-1])/2)
plt.plot(center_x, hist.save_error[0], label=r'$error_ baf$')
plt.xlabel("x")
plt.ylabel("error = exact - computed")
plt.show()
np.save(f'{image_save}/error_tau={tau}', hist.save_error[0])

print(f'Plots saved in {image_root}')