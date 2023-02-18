import numpy as np
import matplotlib.pyplot as plt
import baf
import os

image_root = "/tmp/back_and_forth/"
os.makedirs(image_root, exist_ok = True)

N = 256
h = 1 / N
sigma = 2000.

x = np.linspace(0.5 * h, 1 - 0.5 * h, N)
x, y = np.meshgrid(x, x)

mu = np.exp(-((x - 0.25) ** 2 + (y - 0.25)**2)/ 0.1**2)
# mu = np.zeros_like(x)
# mu[(x - 0.25)**2 + (y - 0.25)**2 < 0.15**2] = 1.

mu /= np.sum(mu)

#nu = np.exp(-((x - 0.65) ** 2 + (y - 0.85)**2)/ 0.1**2)
nu = np.zeros_like(x)
nu[(x - 0.75)**2 + (y - 0.75)**2 < 0.15**2] = 1.
nu[(x - 0.75)**2 + (y - 0.75)**2 < 0.1**2] = 0.
# nu[abs(x - 0.75) + 3 * abs(y - 0.75) < 0.1] = 1.

nu /= np.sum(nu)
#plt.matshow(nu);

# discrete -lap
def nlap(u):
    h = 1 / len(u)
    return 1 / h**2 * (4 * u[1:-1,1:-1] - u[2:,1:-1] - u[:-2,1:-1] - u[1:-1,2:] - u[1:-1,:-2])

# solve -lap v = u with Neumann boundary condition, assuming that nodes are in the centers of the cells
# Solution has mean 0
def nlap_solve(u):
    l = len(u)
    h = 1 / len(u)
    # even extension to size 2Nx2N: enforces Neumann boundary condition
    u = np.concatenate((u, np.flipud(u)))
    u = np.concatenate((u, np.fliplr(u)), axis=1)

    n = len(u)
    m = np.arange(n)
    m, p = np.meshgrid(m, m)

    # discrete -lap acting on Fourier modes
    ker = 2. / h**2 * (2 - np.cos(2 * np.pi * m / n) - np.cos(2 * np.pi * p / n))
    ker[0,0] = 1.  # This is zero in the above formula
    ker_rcp = 1. / ker
    # mean 0 solution
    ker_rcp[0, 0] = 0.
    # technically we need only rfft2 since the input is real valued
    return np.real(np.fft.ifft2(np.fft.fft2(u) * ker_rcp)[:l, :l])

# v = nlap_solve(u)
# plt.matshow(v);
# np.max(np.abs(nlap(v) - u[1:-1,1:-1] + np.mean(u)))

# Wasserstein distance \int \phi d\nu + \int \phi^c d\mu
def w2(phi, psi, mu, nu):
    return np.sum(phi * nu + psi * mu)

phi = np.zeros_like(mu)
psi = np.zeros_like(mu)

temp = np.zeros_like(phi)
pfwd = np.zeros_like(phi)

scaleDown = 0.5
scaleUp   = 1/scaleDown
upper = 0.75
lower = 0.25

def update_sigma(diff, H1_sq, sigma):
    if diff < 0.:
        sigma *= 0.1
    elif diff > H1_sq * sigma * upper:
        sigma *= scaleUp
    elif diff < H1_sq * sigma * lower:
        sigma *= scaleDown
    return sigma

# ascent step of J(phi) = \int phi dnu + \int phi^c dmu
# fills phi and phi_c, returns new sigma
def ascent(phi, phi_c, mu, nu, sigma):
    phi_c[:] = phi
    baf.ctransform2(phi_c, h)

    old_J = w2(phi, phi_c, mu, nu)

    baf.push_forward2(pfwd, mu, phi_c, h)
    rho = nu - pfwd
    lp = nlap_solve(rho)
    phi += sigma * lp

    phi_c[:] = phi
    baf.ctransform2(phi_c, h)

    J = w2(phi, phi_c, mu, nu)
    H1_sq = np.sum(rho * lp)
    return update_sigma(J - old_J, H1_sq, sigma), J, H1_sq


class Hist: pass

hist = Hist()

hist.sigma1 = []
hist.sigma2 = []
hist.J = []
hist.H1_sq = []
hist.phi = []
hist.Tphi_mu = []
hist.Tpsi_nu = []

sigma1 = sigma
sigma2 = sigma
J = 0
H1_sq = 0
common_sigma = False
for i in range(100):

    sigma1, J, H1_sq = ascent(phi, psi, mu, nu, sigma1)
    if common_sigma:
        sigma2 = sigma1

    print(f'J(φ) = {J}, (H¹)² = {H1_sq}, σ₁ = {sigma1}')
    hist.J.append(J)
    hist.H1_sq.append(H1_sq)
    hist.sigma1.append(sigma1)
    hist.phi.append(np.float32(phi))

    baf.push_forward2(pfwd, mu, psi, h)
    hist.Tphi_mu.append(np.float32(pfwd))
    plt.matshow(pfwd)
    plt.savefig(f'{image_root}Tphi_mu{i:04}.png')
    plt.close()

    sigma2, J, H1_sq = ascent(psi, phi, nu, mu, sigma2)
    if common_sigma:
        sigma1 = sigma2

    print(f'I(ψ) = {J}, (H¹)² = {H1_sq}, σ₂ = {sigma2}')
    hist.J.append(J)
    hist.H1_sq.append(H1_sq)
    hist.sigma2.append(sigma2)

    baf.push_forward2(pfwd, nu, phi, h)
    hist.Tpsi_nu.append(np.float32(pfwd))
    plt.matshow(pfwd)
    plt.savefig(f'{image_root}Tpsi_nu{i:04}.png')
    plt.close()

# history plots
plt.plot(hist.sigma1)
plt.plot(hist.sigma2)
plt.savefig(f'{image_root}_sigma.png')
plt.close()

J = np.array(hist.J)
plt.semilogy(np.max(J) - J)
plt.savefig(f'{image_root}_J.png')
plt.close()

plt.semilogy(hist.H1_sq)
plt.savefig(f'{image_root}_H1_sq.png')
plt.close()

np.savez_compressed(f'{image_root}phi', *hist.phi)
np.savez_compressed(f'{image_root}Tphi_mu', *hist.Tphi_mu)
np.savez_compressed(f'{image_root}Tpsi_nu', *hist.Tpsi_nu)

print(f'Plots saved in {image_root}')
