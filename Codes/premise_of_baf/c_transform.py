import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from premise_of_baf.legendre_fenchel import legendre_fenchel

@njit
def c_transform(x, phi, p):
    """costが c(x,p) = |x - p|^2 / 2 である時の、φ phi = φ(y)の c-transform "ψ(psi) = φ^c(p) = |p|^2/2 - φ(p)" を計算する.
    c-transform の値と、 legendre fenchel transform で sup(max) となる時の 座標の添字を return する.
    
     φ^c(p) = inf_x( c(x, p) -  φ(x))              y = f(x) = φ(x) 
            = inf_x( |x - p|^2 / 2 -  φ(x))
            = |p|^2 / 2 - sup_x( xp - (|x|^2 / 2 - φ(x)) ) 
            = |p|^2 / 2 - sup_x( xp - ψ(x) )        ψ(x) = |x|^2/2 - φ(x)       ①
            = |p|^2 / 2 - f^*(p)                                            ②
    """
    #p = x
    psi = 0.5 * x * x - phi                                  # ①
    t, index  = legendre_fenchel(x, psi, p)  #t = f^*(p)       ②
    
    return 0.5 * p * p - np.array(t), index  # phi^c(p) = |p|^2 / 2 - f^*(p)

if __name__ == '__main__':
    
    x = np.array([0, 1])
    y = np.array([0, 1])
    p = np.linspace(-2, 2, 51)
    phi_c, iopt = c_transform(x, y, p)
    plt.plot(p, phi_c)
    phi_cc, iopt2= c_transform(p, phi_c, p)
    plt.plot(p, phi_cc, '.')
    plt.plot(x, y, 'o')
    plt.show()
    # x = np.array([1, 2, 3, 4, 5])
    # y = np.array([0, 1, -2, 1, 0])

    # plt.plot(x, y, label=r'y = f(x)')
    # p = np.array([-2, -1, 0, 1, 2])
    # #p = np.linspace(-1, 1, 51)
    # phi_c, iopt = c_transform(x, y, p)
    # print(phi_c, iopt)
    # print()
    # print(x[iopt])
    # phi_cc, _= c_transform(p, phi_c, x)
    
    
    # plt.show()
    
    
    
    # sin(0.5 * x) is c-concave so doing c-transform twice should return the original function
    x = np.linspace(-10, 10, 101)         #[-10,10]を100等分(-10,…,10), numpy.linspace(最初の値,最後の値,要素数)
    y = np.sin(0.5 * x)
    p = np.linspace(-10, 10, 101) 
    phi_c, iopt = c_transform(x, y, p)

    print(x[iopt])
    phi_cc, ipot2= c_transform(p, phi_c, x)

    plt.title(r'$\phi^{cc} - \phi$')
    plt.plot(x, phi_cc - y)
    plt.show()

    # testing the discrete version of Jacobs-Leger Lemma 1(ii)
    plt.title('Jacobs-Leger Lemma 1(ii)')
    plt.plot(x[:-1], x[:-1] - (phi_c[1:] - phi_c[:-1]) / (x[1:] - x[:-1]), label=r"$x - (\phi^c)'(x)$")
    plt.plot(x, x[iopt], label=r'x[iopt] = $T_\phi(x)$')
    plt.legend()
    plt.show()
    