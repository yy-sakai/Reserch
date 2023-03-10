import numpy as np
import matplotlib.pyplot as plt

# x: x座標, y: y座標, s: スロープ（接線の傾き）, data: 元のデータポイント
class ConvexHullPoint:
    def __init__(self, x, y, s, data):
        self.x = x
        self.y = y
        self.s = s
        self.data = data
        
    def convex_hull_1d(f, data):
        assert data.shape == f.shape
        hull = []                     #凸包をリスト"hull"に保存
        points = np.ndenumerate(f)
        index, (y, data_point) = next(points)
        s = float('-inf')
        prev = ConvexHullPoint(index, y, s, data_point)

        for index, (y, data_point) in points:
            while prev.s >= s:
                s = (y - prev.y) / (index - prev.x)
                prev = hull.pop()
            hull.append(prev)
            prev = ConvexHullPoint(index, y, s, data_point)

        hull.append(prev)
        return hull
    

# Example data
x = np.linspace(0, 10, 100)
y = np.sin(x)
data = np.random.rand(100)

# Compute convex hull
hull = ConvexHullPoint.convex_hull_1d(y, data)

# Plot original data and convex hull
plt.plot(x, y)
plt.scatter(np.arange(len(data)), data)
plt.scatter([p.x for p in hull], [p.data for p in hull], color='r')
plt.show()