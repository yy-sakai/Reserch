import numpy as np
import matplotlib.pyplot as plt

def convex_hull(x, y):
    # 関数f(x, y)のconvex hullを計算する。(y_i = f(x_i))
    # x座標とy座標を入力すると、convex hullのx座標の添字i(x_i)と傾き（その点と一つ前の点で作る傾き）をreturnする。
    
    l = [(0, -np.inf)]                #np.inf = ∞,  l = [(添字（一番左の点）(x_0)、傾き-∞)]
    
    for i, (nx, ny) in enumerate(zip(x[1:],y[1:])):      #nx:x_i(x座標)Now x, ny:y_i(y座標), zip:(リストやタプルなど）の要素を同時に取得（x,yを同時に取得）
                                                         
                                                         #凸関数の場合、2点でできる傾きは単調増加
        while True:
            pi, pv = l[-1]                               #pi:一つ前の(Previous)「点の添字番号（左から数えた点の番号(x_i)）」pv: 一つ前の「現在の点と前の点で作る直線の傾きv」
            v = (ny - y[pi]) / (nx - x[pi])              #v:現在の点と一つ前の点で作った直線の傾き
            #print('v = ', v)
            if v <= pv:                                  #傾きが一つ前の傾きより小さかったら
                del l[-1]                                #一つ前の点（添字、傾き）を消す
                #print(l)
            else:                                        #傾きがひとつ前の傾きより大きい
                l.append((i+1, v))                       #点の情報（添字、傾き）をlに追加する
                #print(l)
                break
        
    return [j for j, _ in l], [v for _, v in l[1:]]      #lは一つの組に2つ要素。j, _という名前でとってくるが、jのみ返す。vも同様.


x = [1, 2, 3, 4, 5]
y = [0, 1, -2, 1, 0]
x = np.array(x)
y = np.array(y)
plt.plot(x, y, label=r'$f$')

plt.plot([1, 3, 4, 5], [0,-2, 1, 0], ":", label=r'convex hull', linewidth=4)
plt.legend()
plt.show()