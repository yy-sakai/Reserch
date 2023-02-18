import numpy as np
from legendre_fenchel import legendre_fenchel

def c_transform(x, phi):
    """Computes the c-transform of phi with cost c(x,y) = |x - y|^2 / 2,  phi = phi(y), psi= |y|^2/2 - phi.

    Returns the c-tranform and the indices of the point where the max is attained.
    """
    psi = 0.5 * x * x - phi
    t, index  = legendre_fenchel(x, psi, x)  #t = f^*(x)
    
    return 0.5 * x * x - t, index  # phi^c(x) = |x|^2 / 2 - f^*(x)

if __name__ == '__main__':
    # sin(0.5 * x) is c-concave so doing c-transform once should return the original function
    import matplotlib.pyplot as plt
    x = np.linspace(-10, 10, 101)         #[-10,10]を100等分(-10,…,10), numpy.linspace(最初の値,最後の値,要素数)
    y = np.sin(0.5 * x)
    phi_c, iopt = c_transform(x, y)
    print(phi_c, iopt)
    print(x[iopt])
    phi_cc, _= c_transform(x, phi_c)

    plt.title(r'$\phi^{cc} - \phi$')
    plt.plot(x, phi_cc - y)
    plt.show()

    # testing the discrete version of Jacobs-Leger Lemma 1(ii)
    plt.title('Jacobs-Leger Lemma 1(ii)')
    plt.plot(x[:-1], x[:-1] - (phi_c[1:] - phi_c[:-1]) / (x[1:] - x[:-1]), label=r"$x - (\phi^c)'(x)$")
    plt.plot(x, x[iopt], label=r'x[iopt] = $T_\phi(x)$')
    plt.legend()
    plt.show()