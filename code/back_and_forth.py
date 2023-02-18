import numpy as np
import matplotlib.pyplot as plt
from c_transform import c_transform
from push_forward import push_forward
from push_forward import lap_solve



x = np.linspace(-1, 1, 101)
p = x

#mu = np.exp(-(x - 0.5)**2 * 100)   #e^(-(x-0.5)^2 * 100)
mu = np.where(x > 0.2, 1., 0.)     #True: 1. False: 0.
mu /= np.sum(mu)                   # mu = mu / np.sum(mu)
#nu = np.copy(mu[::-1])
#nu = np.exp(-(x + 0.2)**2 * 100) + np.exp(-(x+0.7)**2 * 100)
nu = np.exp(-(x + 0.5)**2 * 100)
nu /= np.sum(nu)                    
#plt.plot(x, mu)
#plt.plot(x, nu)

phi = np.zeros_like(x)
psi = np.zeros_like(x)

sigma = 8 / np.max(mu) / 4

phi_iopt = np.arange(len(x))
psi_iopt = np.arange(len(x))

for k in range(20):
    
    #plt.plot(x, lap_solve(nu - push_forward(mu, phi_iopt)))
    #plt.plot(x, phi)
    #plt.show()
    _, phi_iopt = c_transform(x, phi, p)
    
    phi += sigma * lap_solve(nu - push_forward(mu, phi_iopt))
    psi, _ = c_transform(x, phi, p)
    
    _, psi_iopt = c_transform(x, psi, p)
    psi += sigma * lap_solve(mu - push_forward(nu, psi_iopt))
    phi, _ = c_transform(x, psi, p)
    # plt.plot(x, phi)
    # plt.plot(x, psi)
    
    if k % 1 == 0:
        #plt.plot(x, push_forward(mu, phi_iopt))
        """2通りの表現があるT_{phi}の差を調べる"""
        t, phi_iopt = c_transform(x, phi, p)
        # plt.plot(x, x[phi_iopt])
        # plt.plot(x[1:], x[1:] - (t[1:] - t[:-1])/(x[1:] - x[:-1]))
        #plt.plot(x[1:], x[1:] - (t[1:] - t[:-1])/(x[1:] - x[:-1]) - x[phi_iopt][1:])
        """なめらかにする"""
        #plt.plot(x, lap_solve(nu - push_forward(mu, phi_iopt)))
        #plt.plot(x, lap_solve(mu - push_forward(nu, psi_iopt)))
        plt.show()