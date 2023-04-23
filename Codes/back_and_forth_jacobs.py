import numpy as np
import matplotlib.pyplot as plt
from c_transform import c_transform
from jacobs import push_forward1
from push_forward import lap_solve


x = np.linspace(-1, 1, 101)
p = x

mu = np.where((x > 0.) & (x < 0.5), 1, 0.)     #True: 1. False: 0.
nu = np.where((x > -0.5) & (x < -0.), 1., 0.)

plt.plot(x, mu)
plt.plot(x, nu)
plt.show()

phi = np.zeros_like(x)
psi = np.zeros_like(x)

sigma = 1

phi_iopt = np.arange(len(x))
psi_iopt = np.arange(len(x))


h = x[1] - x[0]
plt.title(r'back-and-forth update $\mu$ and $\nu$. Example 2:  Iterate 0')
plt.plot(x, mu,label=r'$\mu$')
plt.plot(x, nu,label=r'$\nu$')
plt.xlim(-1,1)
plt.ylim(-0.5,2.5) 
plt.legend(prop={'size': 15})
plt.show()

for k in range(50):
    phi_c, phi_iopt = c_transform(x, phi, p)
    phi += sigma * lap_solve(nu - push_forward1(mu, phi_c, h))
    psi, _ = c_transform(x, phi, p)                             #psi_{n + 1/2} = (phi_{n + 1/2})^c
    
    psi_c, psi_iopt = c_transform(x, psi, p)
    psi += sigma * lap_solve(mu - push_forward1(nu, psi_c, h))
    phi, _ = c_transform(x, psi, p)                             #phi_{n + 1} = (psi_{n + 1})^c
    
    plt.title(r'back-and-forth update $\mu$ and $\nu$. Example 2:  Iterate ' + str(k+1))
    plt.plot(x, push_forward1(nu, psi_c, h),label=r'$T_{\phi \#} \mu$')
    plt.plot(x, push_forward1(mu, phi_c, h),label=r'$T_{\psi \#} \nu$')
    
    if k % 1 == 0:
        plt.xlim(-1,1)
        plt.ylim(-0.5,2.5) 
        plt.legend(prop={'size': 15})
        plt.show()