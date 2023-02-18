import numpy as np
from convex_hull import convex_hull

def legendre_fenchel(x, y, p):
    """Computes Legendre-Fenchel transform of the given function f given by x and y coords.

    Returns f^*(p) and the corresponding indices i that maximize x[i] p - y[i].
    (f^*(p)=sup_x(px - f(x)) = inf_x(f(x)-px), 傾きpを固定し,その傾きがある点x[i]での∇f(x[i])となるx[i]を探す)
    """
    #print(p)                                      #x = p: x座標を分割した添字 
    chi, s = convex_hull(x, y)                    #chi:convex_hullになったx,y座標の添字(conv) s:現在と次の点を結んだ線の傾き(現在の点での接戦の傾き（微分）)
    #print(chi, s)
    s.append(np.Inf)

    t = []
    iopt = np.zeros_like(p, dtype=int)           #x = p = np.linspace(-10, 10, 100),  np.zeros_like():要素全て0に初期化
    i = 0
    for j, p in enumerate(p):
        while p > s[i]:                  #s[i]:現在と次の点を結んだ線の傾きは単調増加より、p:[-10,10]を99等分(-10,…,10)それを大きくしていき、
            #print(p, s[i])              #傾きpがs[i](分割を小さく取れば∇f(x)と等しい)と等しいもしくは大きくなる時までpを大きくする
            i += 1
        iopt[j] = chi[i]                 
        t.append(x[chi[i]] * p - y[chi[i]])  #p=s[i]=∇f(x)となるx座標=ipot[j]とf^*(p)=tを保存
        
    return t, iopt

if __name__ == '__main__':
    # Legendre-Fenchel is an identity for 0.5 x**2
    x = np.linspace(-10, 10, 100)          #[-10,10]を99等分(-10,…,10), numpy.linspace(最初の値,最後の値,要素数)  
    y = 0.5*x**2

    p = x
    t, _ = legendre_fenchel(x, y, p)      #legendre_fenchelの関数の帰りから、tだけをとってくる
    #print(t, _)
    #print(np.abs(t - y))
    assert np.max(np.abs(t - y)) == 0.        #assert 条件式, 条件式がFalseの場合に出力するメッセージ　（条件式がTrueではない時に、例外を投げる）

    # L-F of double well
    import matplotlib.pyplot as plt
    #plt.plot(x, y)
    #plt.plot(p, t)
    #plt.show()
    
    
    x = np.linspace(-2, 2, 1001)
    y = (x - 1)**2 * (x + 1)**2

    plt.plot(x, y)
    #plt.show()


    p = x
    t, _ = legendre_fenchel(x, y, p)
    #print(t, _)
    plt.plot(p, t)
    #plt.show()

    # f** gives the convex hull
    yss, _ = legendre_fenchel(p, t, x)
    plt.plot(x, yss)
    plt.show()
    