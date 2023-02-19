import numpy as np
import matplotlib.pyplot as plt
from c_transform import c_transform

def push_forward(mu, t):
    """t[i] is the index where t maps index i"""
    
    nu = np.zeros_like(mu)
    for i, t in enumerate(t):
        nu[t] += mu[i]
    return nu

def lap_solve(f):
    """Solves (-\Delta)u = f with Neumann boundary condition on [0,1]. 
    f needs to be given at all nodes including the endpoints. The mean of f is set to zero."""
    # even periodic extension to get cosine series; imaginary part of the result will be zero
    pf = np.concatenate((f, f[-2:0:-1]))
    ff = np.fft.rfft(pf)
    xi = np.linspace(0, 1, len(f))
    N = len(f) - 1
    ff[0] = 0 # set mean to 0
    ff[1:] /= 4 * np.sin(0.5 * np.pi * xi[1:])**2 * N**2
    # perform inverse fft and remove the even periodic extension
    return np.fft.irfft(ff)[:len(f)]


if __name__ == '__main__':
    x = np.linspace(-10, 10, 101)
    p = x
    #y = np.random.random(len(x))
    y = np.sin(0.5 * x)
    y = 0.5 * x * x
    #y = 0 * x

    yy, _ = c_transform(x, y, p)
    t, iopt = c_transform(p, yy, p)
    print(iopt)


    mu = np.ones_like(x)
    nu = push_forward(mu, iopt)

    plt.plot(x, mu)
    plt.plot(x, nu)
    plt.show() 

    plt.plot(x, lap_solve(nu - mu))
    plt.show()