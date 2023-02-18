import numpy as np

def convex_hull(x, y):
    """Computes the convex hull of points with coords `x` and `y`.

    Returns the indices of the points in the hull and the slopes. 
    """

    # index and the slope between this point and the previous
    l = [(0, -np.Inf)]                                   #np.inf = ∞,  lの中身：[(添字（一番左の点）(x_0)、傾き-∞)]
    
    for j, (cx, cy) in enumerate(zip(x[1:],y[1:])):      #cx:x_n(x座標), cy:y_n(y座標), zip:(リストやタプルなど）の要素をnp同時に取得（x,yを同時に取得）
        while True:
            pi, ps = l[-1]                               #pi:一つ前の「点の添字番号（左から数えた点の番号(x_n)）」ps: 一つ前の「現在の点と前の点で作る直線の傾き」
            s = (cy - y[pi]) / (cx - x[pi])              #s:現在の点と一つ前の点で作った直線の傾き
            #print('s = ', s)
            if s <= ps:                                  #傾きが一つ前の傾きより小さかったら
                del l[-1]                                #一つ前の点（添字、傾き）を消す
                #print(l)
            else:                                        #傾きがひとつ前の傾きより大きい
                l.append((j+1, s))                       #点の情報（添字、傾き）をlに追加する
                #print(l)
                break
        
    return [i for i, _ in l], [s for _, s in l[1:]]      #lは一つの組に2つ要素。i, _という名前でとってくるが、iのみ返す。sも同様


if __name__ == '__main__':
    print(convex_hull([1, 2, 3, 4, 5], [0, 1, -2, 1, 0]))      #([0, 2, 4], [-1.0, 1.0])
    
    # plot a convex hull of double well
    import matplotlib.pyplot as plt
    x = np.linspace(-2, 2, 101)                          #[-2,2]を100等分(-2,…,2), numpy.linspace(最初の値,最後の値,要素数)
    y = (x - 1)**2 * (x + 1)**2

    plt.plot(x, y)

    i, _ = convex_hull(x, y)              #xだけ取得
    plt.plot(x[i], y[i])
    plt.show()


    # plot a convex hull of more complicated function
    import matplotlib.pyplot as plt
    x = np.linspace(0, 10, 10000)
    y = np.sin(10 * x) * x**0.5

    plt.plot(x, y)

    i, _ = convex_hull(x, y)
    plt.plot(x[i], y[i])
    plt.show()
