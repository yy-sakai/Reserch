import numpy as np
import matplotlib.pyplot as plt
import os
import time
from c_transform import c_transform
from push_forward_jacobian import push_forward2
from push_forward_jacobian import lap_solve_modified

image_root = "/Users/sakaiyukito/Downloads/LABO/images/back_and_forth_jacobi_test3/"
os.makedirs(image_root, exist_ok = True)

# Wasserstein distance \int \phi d\nu + \int \phi^c d\mu
def w2(phi, psi, mu, nu):
    return np.sum(phi * nu + psi * mu)

# ascent step of J(phi) = \int phi dnu + \int phi^c dmu
# fills phi and phi_c, returns new sigma
# using Jacobian push fofward (push_forward2) 
# common ascent scheme
def ascent(phi, phi_c, mu, nu):
    phi_c, _ = c_transform(x, tau * phi, x)                        # 1-1  phi_c, _ = c_transform(x, phi, p)
    phi_c /= tau

    nmu = max(abs(mu))
    theta_1 = 1 / (2 * gamma)
    theta_2 = tau * nmu
    
    pfwd = push_forward2(mu, tau * phi, h)           # 1-2-1     pfwd : T_{\phi\#}\mu = \mu(x - \tau \nabla \phi(x))|det(I - \tau D^2\phi_c))|
    rho = nu - pfwd                                 # 1-2-2     rho = \nu - T_{\phi\#}\mu　＝ \delta U^*(- \phi) - T_{\phi\#}\mu
    # TODO: This is by far the slowest part of the algorithm
    lp = lap_solve_modified(rho, theta_1, theta_2)                             # 1-2-3     lp: \nabla_{\dot{H}^1} J(\phi_n) = (- \Delta)^{-1} * rho
    phi += lp                               # 1-2-4   phi_{n + 1/2} = phi_n + sigma * lp
#####################################################################                     
    phi_c, _ = c_transform(x, tau * phi, x)                    # 2    psi_{n + 1/2} = (phi_{n + 1/2})^c
    phi_c /= tau
    H1_sq = np.mean(rho * lp)                        #######       ?
    return H1_sq, phi, phi_c, pfwd   #  ?

# whether to plot all timesteps and save the time step data
track = True
H1_sq = 0


x = np.linspace(-0.5, 0.5, 257)
h = x[1] - x[0]


# Set parameters
m = 2
c = np.zeros_like(x)
tau = 0.025
eps = 1e-3             #1.0**(-3)
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
hist.Tphi_mu = []
hist.Tpsi_nu = []

# JKO scheme

start = time.process_time()
timestep = np.arange(0, 2, tau)
stepsize = timestep[1] - timestep[0]

for real_t in timestep:
    diff = 1
    count = 0
# The back-and-forth scheme for solving J(phi) and I(psi)
    while diff >= eps:
        if count > 200:
            break
        if real_t == 0 and count == 0:
            plt.ylim([-0.1, 15.1])
            plt.title(r'back-and-forth update $\mu$ and $\nu$. Example 1:  Iterate ' + str(count))
            plt.plot(x, mu,label=r'$\mu$')
            plt.plot(x, nu,label=r'$\nu$')
            #plt.plot(x, phi,label=r'$\phi$')
            u = np.maximum(1 / t**(1 / 3) * (b - (1 / (12 * t**(2 / 3))) * x**2), 0)
            plt.plot(x, u, "--",label=r'exact')
            plt.legend(prop={'size': 15})
            plt.savefig(f'{image_root}Tphi_mu,Tpsi_nu{real_t:.2}.png', )
            plt.close()
            
            plt.plot(x, phi,label=r'$\phi$')
            plt.legend()
            plt.savefig(f'{image_root}phi{count:04}.png', )
            plt.close()
            error = sum(abs((u - nu)[1:] + (u - nu)[:-1]) * h / 2)  #error = (2 / \tau) * \sigma_{n=0}^{2 / \tau} \int |\rho(n*\tau + t0, x) - \rho^(n)(x)| dx
            #print('error = ', error)
            
            
        nu = (((m - 1) / (m * gamma)) * np.maximum(c - phi, 0)) ** (1 / (m - 1)) # \rho_*(x) = \delta U^*(- \phi) 
        H1_sq, phi, psi, pfwd  = ascent(phi, psi, mu, nu)  # phi = phi_{k + 1/2}, psi = psi_{k + 1/2}
        
        
        # Calculate residual $||\nabla U^*(- \varphi) - T_{\phi \#} \mu||_{L^1(\Omega)}
        #L1 norm
        diff = sum(abs(nu - pfwd) * h)
        
        print(f'{real_t + stepsize:.4}:(H¹)² = {H1_sq:.3}, diff = {diff:.5}, baf_roop = {count}')
            
        hist.H1_sq.append(H1_sq)
        #hist.phi.append(np.float32(phi))
        hist.Tphi_mu.append(np.float32(pfwd))
        
        ##################################################################################
        
        psi_c, _ = c_transform(x, tau * psi, x)        # 3-1  phi_c, _ = c_transform(x, phi, p)
        psi_c /= tau
        
        nu = (((m - 1)/ (m * gamma)) * np.maximum(c - psi_c, 0)) ** (1 / (m - 1)) # nu = T_{\psi \#} \delta U^* (- \psi^c)
        H1_sq, psi, phi, pfwd = ascent(psi, phi, nu, mu)
        
        print(f'{real_t + stepsize:.4}:(H¹)² = {H1_sq:.3}, diff = {diff:.5}, baf_roop = {count}')
        
        hist.H1_sq.append(H1_sq)
        hist.Tpsi_nu.append(np.float32(pfwd))
        hist.phi.append(np.float32(phi))
    
        count+= 1
        
    """"""
    mu = (((m - 1) / (m * gamma)) * np.maximum(c - phi, 0)) ** (1 / (m - 1)) # 
    print(f'Elapsed {time.process_time() - start:.3}s')
    

    plt.ylim([-0.1, 15.1])
    plt.plot(x, nu,label=r'$\nu$')
    
    
    t = (real_t + stepsize + t0) * gamma
    u = np.maximum(1 / t**(1 / 3) * (b - (1 / (12 * t**(2 / 3))) * x**2), 0)
    area = sum((u[1:] + u[:-1]) * h / 2)   #trapezoidal formula
    print(area)
    plt.plot(x, u, "--", label=r'exact')    
    
    
    error += sum(abs((u - nu)[1:] + (u - nu)[:-1]) * h / 2)  #error = (2 / \tau) * \sigma_{n=0}^{2 / \tau} \int |\rho(n*\tau + t0, x) - \rho^(n)(x)| dx
    #print('error = ', error)
    #plt.plot(x, phi,label=r'$\phi$')
    plt.legend(prop={'size': 15})
    plt.savefig(f'{image_root}Tphi_mu,Tpsi_nu{real_t + stepsize:.4}.png', )
    plt.close()


error /= 2 / tau  #error = (2 / \tau) * \sigma_{n=0}^{2 / \tau} \int |\rho(n*\tau + t0, x) - \rho^(n)(x)| dx
print('error = ', error)
plt.semilogy(hist.H1_sq)
plt.savefig(f'{image_root}_H1_sq.png')
plt.close()

print(f'Plots saved in {image_root}')