import numpy as np
import matplotlib.pyplot as plt
import os
from premise_of_baf.c_transform import c_transform
from premise_of_baf.push_forward_jacobian import push_forward2
from premise_of_baf.push_forward_jacobian import lap_solve

image_root = "../images/baf_testtau/"
os.makedirs(image_root, exist_ok = True)

# Wasserstein distance \int \phi d\nu + \int \phi^c d\mu
def w2(phi, psi, mu, nu):
    return np.sum(phi * nu + psi * mu)


#  every ascent step
def update_sigma(diff, H1_sq, sigma):
    """
    if diff < 0.:
        sigma *= 0.1
    elif diff > H1_sq * sigma * upper:
        sigma *= scaleUp
    elif diff < H1_sq * sigma * lower:
        sigma *= scaleDown
    return sigma
    """
    sigma = 1.
    return sigma


# ascent step of J(phi) = \int phi dnu + \int phi^c dmu
# fills phi and phi_c, returns new sigma
# different ascent scheme
def ascent1(phi, phi_c, mu, nu, sigma):
    phi_c, _ = c_transform(x, tau * phi, x)                        # 1-1  phi_c, _ = c_transform(x, phi, p)
    phi_c /= tau
    nu = (((m - 1) / m) * np.maximum(c - phi, 0)) ** (1 / (m - 1)) # \rho_*(x) = \delta U^*(- \phi) 

    pfwd = push_forward2(mu, tau * phi, h)           # 1-2-1     pfwd : T_{\phi\#}\mu = \mu(x - \tau \nabla \phi(x))|det(I - \tau D^2\phi_c))|
    rho = nu - pfwd                                 # 1-2-2     rho = \nu - T_{\phi\#}\mu　＝ \delta U^*(- \phi) - T_{\phi\#}\mu
    # TODO: This is by far the slowest part of the algorithm
    lp = lap_solve(rho)                             # 1-2-3     lp: \nabla_{\dot{H}^1} J(\phi_n) = (- \Delta)^{-1} * rho
    phi += sigma * lp                               # 1-2-4   phi_{n + 1/2} = phi_n + sigma * lp
#####################################################################                              
    phi_c, _ = c_transform(x, tau * phi, x)                    # 2    psi_{n + 1/2} = (phi_{n + 1/2})^c
    phi_c /= tau
    H1_sq = np.mean(rho * lp)                        #######       ?
    return H1_sq, phi, phi_c, pfwd   #  ?


###########################################################################################

def ascent2(psi, psi_c, nu, mu, sigma):
    psi_c, _ = c_transform(x, tau * psi, x)        # 3-1  phi_c, _ = c_transform(x, phi, p)
    psi_c /= tau
    nu = (((m - 1)/ m) * np.maximum(c - psi_c, 0)) ** (1 / (m - 1)) # nu = T_{\psi \#} \delta U^* (- \psi^c)

    pfwd = push_forward2(nu, tau * psi, h)              # 3-2-1     pfwd :T_{\psi\#}\nu = T_{\psi \#} \delta U^* (- \psi^c) \nu = \nu(x - \tau \nabla psi(x))|det(I - \tau D^2\psi_c))|
    rho = mu - pfwd                                 # 3-2-2     rho = \nu - T_{\phi\#}\mu
    # TODO: This is by far the slowest part of the algorithm
    lp = lap_solve(rho)                             # 3-2-3     lp: \nabla_{\dot{H}^1} J(\phi_n) = (- \Delta)^{-1} * rho
    psi += sigma * lp                               # 3-2-4   psi_{n + 1} = psi_n+1/2 + sigma * lp
#####################################################################                            
    psi_c, _ = c_transform(x, tau * psi, x)                    # 4    phi_{n + 1} = (psi_{n + 1})^c
    psi_c /= tau 

    H1_sq = np.mean(rho * lp)                        #######       ?
    return H1_sq, psi, psi_c, pfwd   #  ?


H1_sq = 0

x = np.linspace(-1, 1, 65)
p = x
nu = np.exp(-(x)**2 * 100) #e^(-(x)^2 * 100)    #True: 1. False: 0.
nu_0 = nu
#nu /= np.sum(nu) #　 rho = rho / np.sum(rho)
mu = nu
m = 2
h = x[1] - x[0]
tau = 0.001
c = np.zeros_like(x)

# U(\rho) = 1 / m-1 \int \rho^m dx 
phi = -(m / (m - 1)) * nu ** (m - 1)  #\phi_0 = \phi^(0) = -\delta U(\nu^(0)) =  \delta U(\rho^(0))
psi = np.zeros_like(nu)  
sigma = 100

class Hist: pass

hist = Hist()

hist.H1_sq = []
hist.phi = []
hist.phiu = []
hist.rho = []
hist.Tphi_mu = []
hist.Tpsi_nu = []


for k in range(100):
    if k == 0:
        plt.title(r'back-and-forth update $\mu$ and $\nu$. Example 1:  Iterate ' + str(k))
        plt.plot(x, mu,label=r'$\mu$')
        plt.plot(x, nu,label=r'$\nu$')
        plt.plot(x, phi,label=r'$\phi$')
        plt.plot(x, nu_0 +  tau * 200 * np.exp(-(x)**2 * 100) * (200 * x**2 - 1), label=r'appr')
        plt.legend(prop={'size': 15})
        plt.savefig(f'{image_root}Tphi_mu,Tpsi_nu{k:04}.png', )
        plt.close()
        
        plt.plot(x, phi,label=r'$\phi$')
        plt.legend()
        plt.savefig(f'{image_root}phi{k:04}.png', )
        plt.close()
        
    
    H1_sq, phi, psi, pfwd  = ascent1(phi, psi, mu, nu, sigma)  # phi = phi_{k + 1/2}, psi = psi_{k + 1/2}
    
    print(f'{k:3}: (H¹)² = {H1_sq:.3}')
        

    hist.H1_sq.append(H1_sq)
    #hist.phi.append(np.float32(phi))
    hist.Tphi_mu.append(np.float32(pfwd))
    
    
    plt.title(r'back-and-forth update $\mu$ and $\nu$. Example 1:  Iterate ' + str(k+1))
    plt.plot(x, pfwd,label=r'$T_{\phi \#} \mu$')
    
    ##################################################################################
    
    H1_sq, psi, phi, pfwd = ascent2(psi, phi, nu, mu, sigma)
    
    print(f'{k:3}: (H¹)² = {H1_sq:.3}')
    
    hist.H1_sq.append(H1_sq)
    hist.Tpsi_nu.append(np.float32(pfwd))
    hist.phi.append(np.float32(phi))
    
    plt.plot(x, pfwd,label=r'$T_{\psi \#} \nu$')
    plt.plot(x, nu_0 +  tau * 200 * np.exp(-(x)**2 * 100) * (200 * x**2 - 1), label=r'appr')
    plt.legend(prop={'size': 15})
    plt.savefig(f'{image_root}Tphi_mu,Tpsi_nu{k+1:04}.png', )
    plt.close()
    
    plt.plot(x, phi,label=r'$\phi$')
    plt.legend()
    plt.savefig(f'{image_root}phi{k+1:04}.png', )
    plt.close()

plt.semilogy(hist.H1_sq)
plt.savefig(f'{image_root}_H1_sq.png')
plt.close()

np.savez_compressed(f'{image_root}Tphi_mu', *hist.Tphi_mu)
np.savez_compressed(f'{image_root}Tpsi_nu', *hist.Tpsi_nu)

print(f'Plots saved in {image_root}')