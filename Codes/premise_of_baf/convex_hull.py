import numpy as np
import math
from numba import njit
import matplotlib.pyplot as plt

@njit
def convex_hull(x, y):
    # 関数f(x, y)のconvex hullを計算する。(y_i = f(x_i))
    # x座標とy座標を入力すると、convex hullのx座標の添字i(x_i)と傾き（その点と一つ前の点で作る傾き）をreturnする。
    
    l = [(0, -np.inf)]                #np.inf = ∞,  l = [(添字（一番左の点）(x_0)、傾き-∞)]
    
    for i, (nx, ny) in enumerate(zip(x[1:],y[1:])):      #nx:x_i(x座標)Now x, ny:y_i(y座標), zip:(リストやタプルなど）の要素を同時に取得（x,yを同時に取得）
                                                         
                                                         #凸関数の場合、2点でできる傾きは単調増加
        while True:
            pi, pv = l[-1]                               #pi:一つ前の(Previous)「点の添字番号（左から数えた点の番号(x_i)）」pv: 一つ前の「現在の点と前の点で作る直線の傾きv」
            v = (ny - y[pi]) / (nx - x[pi])              #v:現在の点と一つ前の点で作った直線の傾き
            # print('v = ', v)
            if v <= pv:                                  #傾きが一つ前の傾きより小さかったら
                del l[-1]                                #一つ前の点（添字、傾き）を消す
                #print(l)
            else:                                        #傾きがひとつ前の傾きより大きい
                l.append((i+1, v))                       #点の情報（添字、傾き）をlに追加する
                #print(l)
                break
        
    return [j for j, _ in l], [v for _, v in l[1:]]      #lは一つの組に2つ要素。j, _という名前でとってくるが、jのみ返す。vも同様.


if __name__ == '__main__':
    print(convex_hull([1, 2, 3, 4, 5], [0, 1, -2, 1, 0]))      #([0, 2, 4], [-1.0, 1.0])
    
    x = [1, 2, 3, 4, 5]
    y = [0, 1, -2, 1, 0]

    plt.plot(x, y)
    
    x = np.array(x)
    y = np.array(y)
    j, _ = convex_hull(x, y)
    print(x[j], y[j])
    plt.plot(x[j], y[j])
    plt.show()

    
    # plot a convex hull of double well
    
    x = np.linspace(-2, 2, 101)                          #[-2,2]を100等分(-2,…,2), numpy.linspace(最初の値,最後の値,要素数)
    y = (x - 1)**2 * (x + 1)**2

    plt.plot(x, y)

    j, _ = convex_hull(x, y)              #xだけ取得
    plt.plot(x[j], y[j])
    plt.show()


    # plot a convex hull of more complicated function
    x = np.linspace(0, 10, 10000)
    y = np.sin(10 * x) * x**0.5

    plt.plot(x, y)

    j, _ = convex_hull(x, y)
    plt.plot(x[j], y[j])
    plt.show()
    
    
    
    
    x = np.linspace(-1.5, 1.5, 10)
    y = -np.exp(-x**2)

    plt.plot(x, y)
        
    j, _ = convex_hull(x, y)
    plt.plot(x[j], y[j])
    plt.show()
        
    x = np.linspace(-2, 2, 10000)
    y = -np.exp(-x**2)
        
    plt.plot(x, y)
        
    j, _ = convex_hull(x, y)
    plt.plot(x[j], y[j])
    plt.show()
        
    x = np.linspace(-5, 5, 10000)
    y = -math.e**(-x**2)

    plt.plot(x, y)
    j, _ = convex_hull(x, y)
        
    plt.plot(x[j], y[j])
    plt.show()
        
    x = np.linspace(-100, 100, 10000)
    y = -math.e**(-x**2)

    plt.plot(x, y)
        
    j, _ = convex_hull(x, y)
    plt.plot(x[j], y[j])
    plt.show()