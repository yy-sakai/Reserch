import numpy as np
from convex_hull import convex_hull

def legendre_fenchel(x, y, p):
    """Computes Legendre-Fenchel transform of the given function f given by x and y coords.

    Returns f^*(p) and the corresponding indices i that maximize x[i] p - y[i].
    """
    chi, s = convex_hull(x, y)                   #chi:x座標を分割した添字 s:現在と次の点を結んだ線の傾き
    #print(chi, s)
    s.append(np.Inf)

    t = []
    iopt = np.zeros_like(p, dtype=int)           #x = p = np.linspace(-10, 10, 100),  np.zeros_like():要素全て0に初期化
    i = 0
    for j, p in enumerate(p):
        while p > s[i]:
            #print(p, s[i])
            i += 1
        iopt[j] = chi[i]
        t.append(x[chi[i]] * p - y[chi[i]])
        
    return t, iopt

if __name__ == '__main__':
    # Legendre-Fenchel is an identity for 0.5 x**2
    x = np.linspace(-10, 10, 100)          #[-10,10]を20等分(-10,…,10), numpy.linspace(最初の値,最後の値,要素数)  x = [10.0, 9.0,..., 10.0]
    y = 0.5*x**2

    p = x
    t, _ = legendre_fenchel(x, y, p)
    print(t, _)
    assert np.max(np.abs(t - y)) == 0.

    # L-F of double well
    import matplotlib.pyplot as plt
    plt.plot(p, t)
    plt.show()
    
    x = np.linspace(-2, 2, 1001)
    y = (x - 1)**2 * (x + 1)**2

    plt.plot(x, y)
    plt.show()

    p = x
    t, _ = legendre_fenchel(x, y, p)
    plt.plot(p, t)
    plt.show()

    # f** gives the convex hull
    yss, _ = legendre_fenchel(p, t, x)
    plt.plot(x, yss)
    plt.show()