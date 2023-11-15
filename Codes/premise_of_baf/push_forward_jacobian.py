import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from c_transform import c_transform
import numpy as np

@njit
def push_forward2(mu, phi_c, h):
    assert mu.shape == phi_c.shape
    
    t_mu = np.zeros_like(mu)

    n = phi_c.shape[0]
    l_dphi_c = []
    l_det = []
    
    # iterate over each cell
    for i in range(n):
        # 3 neighboring cells
        um = phi_c[max(i-1, 0)]
        u = phi_c[i]
        up = phi_c[min(i+1, n-1)]

        # \nabla\phi_c
        dphi_c = (up - um) / (2. * h)
        #print(dphi_c)
        l_dphi_c.append(dphi_c)
        # det (I - D^2\phi_c)
        det = (1. - (up - 2. * u + um) / (h * h)) 
        #print(det)
        l_det.append(det)
        # x - \nabla\phi_c with respect to the cell grid
        xcell = i - (dphi_c / h)

        # indices of the nearest cells
        ti = int(min(max(np.floor(xcell), 0.), n - 1))
        tio = min(ti + 1, n - 1)

        # interpolate the density value
        mu_inter = mu[ti] * (1. - (xcell - np.floor(xcell))) + mu[tio] * (xcell - np.floor(xcell))

        t_mu[i] = mu_inter * det
        
    return t_mu


"""
def det(det, phi_c, h):
    assert det.shape == phi_c.shape

    n = phi_c.shape[0]

    # iterate over each cell
    for i in range(n):
        # 3 neighboring cells
        um = phi_c[max(i-1, 0)]
        u = phi_c[i]
        up = phi_c[min(i+1, n-1)]

        # det (I - D^2\phi_c)
        det[i] = (1. - (up - 2. * u + um) / (h * h)) ** 2

"""
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


def lap_solve_modified(f, theta_1, theta_2):
    """Solves (\theta1 I - \theta2 \Delta)u = f with Neumann boundary condition on [0,1]. 
    f needs to be given at all nodes including the endpoints. The mean of f is set to zero."""
    # even periodic extension to get cosine series; imaginary part of the result will be zero
    pf = np.concatenate((f, f[-2:0:-1]))
    ff = np.fft.rfft(pf)
    xi = np.linspace(0, 1, len(f))
    N = len(f) - 1
    ff /= (theta_2 * 4 * np.sin(0.5 * np.pi * xi)**2 * N**2) + theta_1  
    # perform inverse fft and remove the even periodic extension
    return np.fft.irfft(ff)[:len(f)]


if __name__ == '__main__':
    
    tau = 1 / 2
    x = np.linspace(-10, 10, 101)
    p = x
    #y = tau * np.random.random(len(x))
    #y = np.sin(x)      #(y = tau * phi)
    y = 0.5 * x * x
    phi = y
    #y = 0 * x
    
    phi_c, _ = c_transform(x, y, p)
    #phi_c *= tau                    # phi_c = (1 / tau) * (tau * phi)_c
    #plt.plot(x, t)     
    #plt.show()  
   
    h = x[1] - x[0]
    mu = np.ones_like(x)
    #mu[np.abs(x) < 1] = 0
    nu = push_forward2(mu, phi_c, h) #phi_c^c = psi
    #nu = push_forward2_tau(mu, phi_c, h, tau) #phi_c^c = psi
    
    plt.plot(x, mu)
    plt.plot(x, nu)
    plt.show() 

    #plt.plot(x, lap_solve(nu - mu))
    #plt.show()



"""
def push_forward2_ver2(t_mu, mu, phi_c, h):
    assert t_mu.shape == mu.shape
    assert t_mu.shape == phi_c.shape

    ni, nj = phi_c.shape

    # map each cell
    for i in range(ni):
        for j in range(nj):
            # 9 neighboring cells
            umm = phi_c[max(i - 1, 0), max(j - 1, 0)]
            um_ = phi_c[max(i - 1, 0), j]
            ump = phi_c[max(i - 1, 0), min(j + 1, nj - 1)]
            u_m = phi_c[i, max(j - 1, 0)]
            u__ = phi_c[i, j]
            u_p = phi_c[i, min(j + 1, nj - 1)]
            upm = phi_c[min(i + 1, ni - 1), max(j - 1, 0)]
            up_ = phi_c[min(i + 1, ni - 1), j]
            upp = phi_c[min(i + 1, ni - 1), min(j + 1, nj - 1)]
            
            # nabla phi_c
            dphi_cx = (up_ - um_) / (2. * h)
            dphi_cy = (u_p - u_m) / (2. * h)
            
            # det (I - D^2 phi_c)
            det = (1. - (up_ - 2. * u__ + um_) / (h * h)) * (1. - (u_p - 2. * u__ + u_m) / (h * h)) - ((upp + umm - ump - upm) / (4. * h * h)) ** 2
            
            # x - nabla phi_c with respect to the cell grid
            xcell = i - dphi_cx / h
            ycell = j - dphi_cy / h
            
            # indices of the nearest 4 cells
            ti = int(max(xcell, 0.)) if xcell >= 0 else 0
            tj = int(max(ycell, 0.)) if ycell >= 0 else 0
            tio = min(ti + 1, ni - 1)
            tjo = min(tj + 1, nj - 1)
            
            a = xcell - int(xcell)
            b = ycell - int(ycell)
            
            # interpolate the density value
            mu_inter = mu[ti, tj] * (1. - a) * (1. - b) + mu[tio, tj] * a * (1. - b) + mu[ti, tjo] * (1. - a) * b + mu[tio, tjo] * a * b
            
            t_mu[i, j] = mu_inter * det

def det(det, phi_c, h):
    assert det.shape == phi_c.shape

    ni, nj = phi_c.shape

    # map each cell
    for i in range(ni):
        for j in range(nj):
            # 9 neighboring cells
            umm = phi_c[max(i - 1, 0), max(j - 1, 0)]
            um_ = phi_c[max(i - 1, 0), j]
            ump = phi_c[max(i - 1, 0), min(j + 1, nj - 1)]
            u_m = phi_c[i, max(j - 1, 0)]
            u__ = phi_c[i, j]
            u_p = phi_c[i, min(j + 1, nj - 1)]
            upm = phi_c[min(i + 1, ni - 1), max(j - 1, 0)]
            up_ = phi_c[min(i + 1, ni - 1), j]
            upp = phi_c[min(i + 1, ni - 1), min(j + 1, nj - 1)]
            
            # det (I - D^2 phi_c)
            det[i, j] = (1. - (up_ - 2. * u__ + um_) / (h * h)) * (1. - (u_p - 2. * u__ + u_m) / (h * h)) - ((upp + umm - ump - upm) / (4. * h * h)) ** 2
            dphi_cx = (up_ - um_) / (2. * h)
            dphi_cy = (u_p - u_m) / (2. * h)
            # det[i, j] = dphi_cy

# 使用例
ni, nj = 10, 10
h = 0.1
phi_c = np.random.rand(ni, nj)
mu = np.random.rand(ni, nj)
t_mu = np.zeros((ni, nj))
det = np.zeros((ni, nj))

push_forward2_ver2(t_mu, mu, phi_c, h)
det(det, phi, h)
"""