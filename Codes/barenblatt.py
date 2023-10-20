import numpy as np
import matplotlib.pyplot as plt

h0 = 15
M = 0.5
b = (np.sqrt(3) * M / 8)**(2 / 3)
gamma = 1e-3
t0 = 1 / gamma * (b / h0)**3
h = 1000
x = np.linspace(-1, 1, h)

for i in np.arange(0, 2, 0.4):
    t = (i + t0) * gamma
    rho = np.maximum(1 / t**(1 / 3) * (b - (1 / (12 * t**(2 / 3))) * x**2), 0)
    area = sum((rho[1:] + rho[:-1]) * (x[1] - x[0]) / 2)
    simpson = sum((rho[:-2] + 4*rho[1:-1] + rho[2:]) / (3 * h))
    simpson2 = sum((rho[:-2] + 4*rho[1:-1] + rho[2:]) * (x[1] - x[0])/ (3*2))
    
    print(area, simpson, simpson2)
    print('area = ',abs(0.5 - area), 'simpson = ', abs(0.5 - simpson), 'simpson2 = ', abs(0.5 - simpson2))
    plt.plot(x, rho)
    
plt.show()