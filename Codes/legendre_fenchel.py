import numpy as np
from convex_hull import convex_hull
import matplotlib.pyplot as plt

def legendre_fenchel(x, y, p):
    #y = f(x)で与えられるinput(x, y)に対し、Legendre-Fenchel transformを計算する
    #(f^*(p)=sup_x(px - f(x)) = inf_x(f(x)-px), 傾きvを固定し,x[i] p - y[i]が最大となるx[i]を探す。すなわち、傾きvがある点x[i]での∇f(x[i])となるx[i]を探す
    
    #print(p)                                      #x = p: x座標を分割した添字 
    chi, v = convex_hull(x, y)                    #chi:convex_hullになったx,y座標の添字(conv) v:現在と次の点を結んだ線の傾き(現在の点での接戦の傾き（微分）)
    #print(chi, v)
    v.append(np.Inf)

    t = []
    iopt = np.zeros_like(p, dtype=int)           #x = v = np.linspace(-10, 10, 100),  np.zeros_like():要素全て0に初期化
    i = 0
    for j, p in enumerate(p):
        while p > v[i]:                  #v[i]:現在と次の点を結んだ線の傾きは単調増加より、p:[-10,10]を99等分(-10,…,10)したものに対し、v[i]^を大きくしていき、
            #print(v, s[i])              #傾きpがv[i](分割を小さく取れば∇f(x)と等しい)と等しいもしくは大きくなる時までvを大きくする
            i += 1
        iopt[j] = chi[i]                 
        t.append(x[chi[i]] * p - y[chi[i]])  #p=v[i]=∇f(x)となるx座標=ipot[j]とf^*(v)=tを保存
        
    return t, iopt

if __name__ == '__main__':
   
    
    x = [1, 2, 3, 4, 5]
    y = [0, 1, -2, 1, 0]

    plt.plot(x, y, label=r'y = f(x)')
    
    p = np.linspace(-1,1,50)
    t, _ = legendre_fenchel(x, y, p)
    
    plt.plot(p, t, 'o', label=r'$f^*(p) = \sup_{x}\{xp - f(x)\}$')
    
    t, iopt = legendre_fenchel(x, y, p)
    print(t, iopt)
    # f** gives the convex hull
    yss, _ = legendre_fenchel(p, t, x)
    plt.plot(x, yss, label=r'$f^{**}(x) = \sup_{p}\{xp - f(p)\}$')
    plt.legend()
    plt.show()
    #-------------------------------------------------------------
    
    # Legendre-Fenchel is an identity for 0.5 x**2
    
    x = np.linspace(-10, 10, 101)          #[-10,10]を100等分(-10,…,10), numpy.linspace(最初の値,最後の値,要素数)  
    y = 0.5*x**2

    p = x
    t, _ = legendre_fenchel(x, y, p)      #legendre_fenchelの関数の帰りから、tだけをとってくる
    #print(t, _)
    #print(np.abs(t - y))
    assert np.max(np.abs(t - y)) == 0.        #assert 条件式, 条件式がFalseの場合に出力するメッセージ　（条件式がTrueではない時に、例外を投げる）
    plt.plot(x, y, label=r'$y = \frac{1}{2}x^2$')
    plt.plot(p, t, label=r'$f^*(p) = \sup_{x}\{xp - f(x)\}$')
    plt.legend()
    plt.show()
    
    #-----------------------------------------------
    
    
    # L-F of double well
    x = np.linspace(-2, 2, 1001)
    y = (x - 1)**2 * (x + 1)**2

    plt.plot(x, y)
    #plt.show()
    p = np.linspace(-1000, 1000, 1001)
    t, _ = legendre_fenchel(x, y, p)
    plt.plot(p, t)

    # f** gives the convex hull
    yss, _ = legendre_fenchel(p, t, x)
    plt.plot(x, yss)
    plt.xlim(-2,2)
    plt.ylim(0,10)
    plt.show()
    