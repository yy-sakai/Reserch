import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
from c_transform import c_transform
from push_forward_jacobi import push_forward2
from push_forward_jacobi import lap_solve

image_root = "/Users/sakaiyukito/Downloads/LABO/images/back_and_forth/"
os.makedirs(image_root, exist_ok = True)

# Wasserstein distance \int \phi d\nu + \int \phi^c d\mu
def w2(phi, psi, mu, nu):
    return np.sum(phi * nu + psi * mu)


#  every ascent step
def update_sigma(diff, H1_sq, sigma):
    pass
    if diff < 0.:
        sigma *= 0.1
    elif diff > H1_sq * sigma * upper:
        sigma *= scaleUp
    elif diff < H1_sq * sigma * lower:
        sigma *= scaleDown
    return sigma


# ascent step of J(phi) = \int phi dnu + \int phi^c dmu
# fills phi and phi_c, returns new sigma
def ascent1(phi, phi_c, mu, nu, sigma):
    #phi_c[:] = phi
    nu = ((m - 1)/ m * np.maximum(c - phi, 0)) ** (1/ (m - 1))
    phi_c, _ = c_transform(x, phi, x)                        # 1-1  phi_c, _ = c_transform(x, phi, p)

    old_J = w2(phi, phi_c, mu, nu)  

    pfwd = push_forward2(mu, phi_c, h)              # 1-2-1     pfwd : T_{\phi\#}\mu = x - (\nabla h)^{-1}(\nabla\phi^c(x)) = x - \nabla\phi^c(x)
    rho = nu - pfwd                                 # 1-2-2     rho = \nu - T_{\phi\#}\mu
    # TODO: This is by far the slowest part of the algorithm
    lp = lap_solve(rho)                             # 1-2-3     lp: \nabla_{\dot{H}^1} J(\phi_n) = (- \Delta)^{-1} * rho
    phi += sigma * lp                               # 1-2-4   phi_{n + 1/2} = phi_n + sigma * lp
#####################################################################
    #phi_c[:] = phi                                 
    phi_c, _ = c_transform(x, phi, x)                    # 2    psi_{n + 1/2} = (phi_{n + 1/2})^c
    J = w2(phi, phi_c, mu, nu)
    H1_sq = np.mean(rho * lp)                        #######       ?
    return J, H1_sq, phi, phi_c, pfwd   #  ?


###########################################################################################

def ascent2(psi, psi_c, mu, nu, sigma):
    #phi_c[:] = phi
    nu = ((m - 1)/ m * np.maximum(c - psi_c, 0)) ** (1/ (m - 1))
    psi_c, _ = c_transform(x, psi, x)                        # 1-1  phi_c, _ = c_transform(x, phi, p)

    old_J = w2(psi, psi_c, mu, nu)  

    pfwd = push_forward2(nu, psi_c, h)              # 1-2-1     pfwd : T_{\phi\#}\mu = x - (\nabla h)^{-1}(\nabla\phi^c(x)) = x - \nabla\phi^c(x)
    rho = mu - pfwd                                 # 1-2-2     rho = \nu - T_{\phi\#}\mu
    # TODO: This is by far the slowest part of the algorithm
    lp = lap_solve(rho)                             # 1-2-3     lp: \nabla_{\dot{H}^1} J(\phi_n) = (- \Delta)^{-1} * rho
    phi += sigma * lp                               # 1-2-4   phi_{n + 1/2} = phi_n + sigma * lp
#####################################################################
    #phi_c[:] = phi                                 
    psi_c, _ = c_transform(x, psi, x)                    # 2    psi_{n + 1/2} = (phi_{n + 1/2})^c
    J = w2(psi, psi_c, mu, nu)
    H1_sq = np.mean(rho * lp)                        #######       ?
    return J, H1_sq, psi, psi_c, pfwd   #  ?



scaleDown = 0.5  # \alpha_2
scaleUp   = 1 / scaleDown # \alpha_1
upper = 0.75     # \beta_2
lower = 0.25     # \beta_1

sigma = 1.
sigma1 = sigma
sigma2 = sigma
J = 0
H1_sq = 0
common_sigma = True




x = np.linspace(-1, 1, 1001)
p = x

rho = np.exp(-(x)**2 * 100)   #e^(-(x-0.5)^2 * 100)    #True: 1. False: 0.
rho /= np.sum(rho)                   # rho = rho / np.sum(rho)
mu = rho

nu = np.exp(-(x + 0.2)**2 * 100) + np.exp(-(x+0.7)**2 * 100)
nu /= np.sum(nu) 

m = 2
c = np.zeros_like(x)
 
phi = -  m / (m - 1) * rho ** (m - 1)


psi = np.zeros_like(x)  
rho_star = np.zeros_like(x)  
dU_m = np.zeros_like(x)
#sigma = 100
#plt.title(r'back-and-forth update $\rho$ and $\mu$. Example 2:  Iterate 0')
plt.plot(x,phi,label=r'$\phi$')
plt.plot(x, rho,label=r'$\rho$')
plt.plot(x, mu,label=r'$\mu$')
plt.legend(prop={'size': 15})
plt.show()

h = x[1] - x[0]


class Hist: pass

hist = Hist()

hist.sigma1 = []
hist.sigma2 = []
hist.J = []
hist.I = []
hist.H1_sq = []
hist.phi = []
hist.Tphi_mu = []
hist.Tpsi_nu = []


for k in range(100):
    plt.title(r'back-and-forth update $\mu$ and $\nu$. Example 1:  Iterate ' + str(k+1))
    #title = ax.text(4.5, 1.15, 'back-and-forth update $\mu$ and $\nu$. Example 2:  Iterate {}'.format(str(k+1)))
    
    J, H1_sq, phi, psi, pfwd  = ascent1(phi, psi, mu, nu,sigma)
    
    print(f'{k:3}: J(φ) = {J}, (H¹)² = {H1_sq:.3}, σ₂ = {sigma1:.5}')
    
    if common_sigma:
        sigma2 = sigma1
        
    hist.J.append(J)
    hist.H1_sq.append(H1_sq)
    hist.sigma1.append(sigma1)
    
    hist.phi.append(np.float32(phi))
    hist.Tphi_mu.append(np.float32(pfwd))
    
    
    
    if k == 0:
        plt.plot(x, nu,label=r'$\nu$')
    else:
        plt.plot(x, pfwd,label=r'$T_{\psi \#} \nu$')
    #img2, = ax.plot(x, pfwd, color='blue', label=r'$T_{\psi \#} \nu$')
    
    
    J, H1_sq, psi, phi, pfwd = ascent2(psi, phi, nu, mu,sigma)
    
    if common_sigma:
        sigma1 = sigma2
    
    print(f'{k:3}: I(ψ) = {J}, (H¹)² = {H1_sq:.3}, σ₂ = {sigma2:.5}')
    
    hist.I.append(J)
    hist.H1_sq.append(H1_sq)
    hist.sigma2.append(sigma2)
    
    hist.Tpsi_nu.append(np.float32(pfwd))

        
    if k == 0:
        plt.plot(x, mu,label=r'$\mu$')
    else:
        plt.plot(x, pfwd,label=r'$T_{\phi \#} \mu$')
        
    
    plt.legend(prop={'size': 15})
    plt.savefig(f'{image_root}Tphi_mu,Tpsi_nu{k:04}.png', )
    plt.close()
    #img1, = ax.plot(x, pfwd, color='red',label=r'$T_{\phi \#} \mu$')
    
    
plt.plot(hist.sigma1)
plt.plot(hist.sigma2)
plt.savefig(f'{image_root}_sigma.png')
plt.close()
    
    
J = np.array(hist.J)
plt.semilogy(np.max(J) - J)
plt.savefig(f'{image_root}_J.png')
plt.close()

I = np.array(hist.I)
plt.semilogy(np.max(I) - I)
plt.savefig(f'{image_root}_I.png')
plt.close()

plt.semilogy(hist.H1_sq)
plt.savefig(f'{image_root}_H1_sq.png')
plt.close()

plt.semilogy(hist.phi)
plt.savefig(f'{image_root}_phi.png')
plt.close()

np.savez_compressed(f'{image_root}phi', *hist.phi)
np.savez_compressed(f'{image_root}Tphi_mu', *hist.Tphi_mu)
np.savez_compressed(f'{image_root}Tpsi_nu', *hist.Tpsi_nu)

print(f'Plots saved in {image_root}')