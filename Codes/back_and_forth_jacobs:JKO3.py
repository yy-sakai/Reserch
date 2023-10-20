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
    #phi_c[:] = phi
    
    phi_c, _ = c_transform(x, tau * phi, x)                        # 1-1  phi_c, _ = c_transform(x, phi, p)
    phi_c /= tau

    nmu = max(np.abs(mu))
    theta_1 = 1 / 2
    theta_2 = tau * nmu
    print(round(tau * nmu, 7), round(tau * (1 + max(-(m / (m - 1)) * nu ** (m - 1))) / 2, 7))
    # if k == 1:
    #     theta_2 = tau * nmu
    # else:
    #     theta_2 = tau * (1 + max(-(m / (m - 1)) * nu ** (m - 1))) / 2
    #     print(tau * nmu, theta_2)

    pfwd = push_forward2(mu, tau * phi, h)           # 1-2-1     pfwd : T_{\phi\#}\mu = \mu(x - \tau \nabla \phi(x))|det(I - \tau D^2\phi_c))|
    rho = nu - pfwd                                 # 1-2-2     rho = \nu - T_{\phi\#}\mu　＝ \delta U^*(- \phi) - T_{\phi\#}\mu
    # TODO: This is by far the slowest part of the algorithm
    lp = lap_solve_modified(rho, theta_1, theta_2)                             # 1-2-3     lp: \nabla_{\dot{H}^1} J(\phi_n) = (- \Delta)^{-1} * rho
    phi += lp                               # 1-2-4   phi_{n + 1/2} = phi_n + sigma * lp
#####################################################################
    #phi_c[:] = phi                                 
    phi_c, _ = c_transform(x, tau * phi, x)                    # 2    psi_{n + 1/2} = (phi_{n + 1/2})^c
    phi_c /= tau
    H1_sq = np.mean(rho * lp)                        #######       ?
    return H1_sq, phi, phi_c, pfwd   #  ?

# whether to plot all timesteps and save the time step data
track = True

H1_sq = 0

x = np.linspace(-1, 1, 65)
p = x
nu = np.exp(-(x)**2*100) #e^(-(x)^2 * 100)    #True: 1. False: 0.
nu_0 = nu
#nu /= np.sum(nu) #　 rho = rho / np.sum(rho)
mu = nu
m = 2
h = x[1] - x[0]
tau = 1e-3
c = np.zeros_like(x)
eps = 1e-3             #1.0**(-3)
diff = 1
M = nu

# U(\rho) = 1 / m-1 \int \rho^m dx 
phi = -(m / (m - 1)) * nu ** (m - 1)  #\phi_0 = \phi^(0) = -\delta U(\nu^(0)) =  \delta U(\rho^(0))
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

for i in range (50):
    j = 0
    diff = 1
    count = 0
# The back-and-forth scheme for solving J(phi) and I(psi)
    while diff >= eps:
        if count > 200:
            break
        if i == 0 and j == 0:
            plt.ylim([0, 1.2])
            plt.title(r'back-and-forth update $\mu$ and $\nu$. Example 1:  Iterate ' + str(j))
            plt.plot(x, mu,label=r'$\mu$')
            plt.plot(x, nu,label=r'$\nu$')
            #plt.plot(x, phi,label=r'$\phi$')
            #plt.plot(x, (np.maximum(((M / (4 * np.pi * m * i))**((m - 1)/ m) - ((m - 1) / (4 * m**2 * i) * x**2))**(1 / (m - 1)), 0)), label=r'appr')
            plt.legend(prop={'size': 15})
            plt.savefig(f'{image_root}Tphi_mu,Tpsi_nu{j:04}.png', )
            plt.close()
            
            plt.plot(x, phi,label=r'$\phi$')
            plt.legend()
            plt.savefig(f'{image_root}phi{j:04}.png', )
            plt.close()
            
        #k = 1
        nu = (((m - 1) / m) * np.maximum(c - phi, 0)) ** (1 / (m - 1)) # \rho_*(x) = \delta U^*(- \phi) 
        H1_sq, phi, psi, pfwd  = ascent(phi, psi, mu, nu)  # phi = phi_{k + 1/2}, psi = psi_{k + 1/2}
        
        
        # Calculate residual $||\nabla U^*(- \varphi) - T_{\phi \#} \mu||_{L^1(\Omega)}
        #L1 norm
        diff = np.sum(np.abs(nu - pfwd)) * h
        
        print(f'{i:3}:(H¹)² = {H1_sq:.3}, diff = {diff:.5}, j = {j}')
            
        hist.H1_sq.append(H1_sq)
        #hist.phi.append(np.float32(phi))
        hist.Tphi_mu.append(np.float32(pfwd))
        
        """
        plt.title(r'back-and-forth update $\mu$ and $\nu$. Example 1:  Iterate ' + str(k+1))
        plt.plot(x, pfwd,label=r'$T_{\phi \#} \mu$')
        """
        
        ##################################################################################
        
        
        psi_c, _ = c_transform(x, tau * psi, x)        # 3-1  phi_c, _ = c_transform(x, phi, p)
        psi_c /= tau
        
        #k = 2
        nu = (((m - 1)/ m) * np.maximum(c - psi_c, 0)) ** (1 / (m - 1)) # nu = T_{\psi \#} \delta U^* (- \psi^c)
        H1_sq, psi, phi, pfwd = ascent(psi, phi, nu, mu)
        
        print(f'{j:3}:(H¹)² = {H1_sq:.3}')
        
        hist.H1_sq.append(H1_sq)
        hist.Tpsi_nu.append(np.float32(pfwd))
        hist.phi.append(np.float32(phi))
        
        """
        plt.plot(x, pfwd,label=r'$T_{\psi \#} \nu$')
        plt.plot(x, nu_0 +  tau * 200 * np.exp(-(x)**2 * 100) * (200 * x**2 - 1), label=r'appr')
        plt.legend(prop={'size': 15})
        plt.savefig(f'{image_root}Tphi_mu,Tpsi_nu{k+1:04}.png', )
        plt.close()
        
        plt.plot(x, phi,label=r'$\phi$')
        #plt.ylim(-0.018, 0.005)
        plt.legend()
        plt.savefig(f'{image_root}phi{k+1:04}.png', )
        plt.close()
        """
    
        j += 1
        count+= 1
        
    """"""
    mu = (((m - 1) / m) * np.maximum(c - phi, 0)) ** (1 / (m - 1)) # 
    print(f'Elapsed {time.process_time() - start:.3}s')
    
    #alpha = 1 / (m-1)
    #beta = 1 / (m+1)
    #gamma = (m-1) / (2*m*(m+1))
    plt.ylim([0, 1.2])
    plt.plot(x, nu,label=r'$\nu$')
    #plt.plot(x, nu,label=r'$\nu$')
    t = i * tau
    if i > 0:
        plt.plot(x, (np.maximum(((M / (4 * np.pi * m * t))**((m - 1)/ m) - ((m - 1) / (4 * m**2 * t) * x**2))**(1 / (m - 1)), 0)), label=r'appr')    #plt.plot(x, np.maximum((1 /(time.process_time() - start)**beta * (M - gamma*(x/(time.process_time() - start)**beta)**2)**alpha), 0), label=r'appr2')
    #plt.plot(x, phi,label=r'$\phi$')
    plt.legend(prop={'size': 15})
    plt.savefig(f'{image_root}Tphi_mu,Tpsi_nu{i+1:04}.png', )
    plt.close()


plt.semilogy(hist.H1_sq)
plt.savefig(f'{image_root}_H1_sq.png')
plt.close()


np.savez_compressed(f'{image_root}Tphi_mu', *hist.Tphi_mu)
np.savez_compressed(f'{image_root}Tpsi_nu', *hist.Tpsi_nu)

print(f'Plots saved in {image_root}')