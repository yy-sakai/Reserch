import numpy as np

def convex_hull(x, y):
    """Computes the convex hull of points with coords `x` and `y`.

    Returns the indices of the points in the hull and the slopes. 
    """

    # index and the slope between this point and the previous
    l = [(0, -np.Inf)]
    
    for j, (cx, cy) in enumerate(zip(x[1:],y[1:])):
        while True:
            pi, ps = l[-1]
            s = (cy - y[pi]) / (cx - x[pi])
            if s <= ps:
                del l[-1]
            else:
                l.append((j+1, s))
                break
        
    
    return [i for i, _ in l], [s for _, s in l[1:]]


if __name__ == '__main__':
    print(convex_hull([1,2,3,4, 5], [0, 1, -2, 1, 0]))

    # plot a convex hull of double well
    import matplotlib.pyplot as plt
    x = np.linspace(-2, 2, 101)
    y = (x - 1)**2 * (x + 1)**2

    plt.plot(x, y)

    i, _ = convex_hull(x, y)
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