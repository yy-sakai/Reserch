import numpy as np
import matplotlib.pyplot as plt

"""
x = np.linspace(-0.5, 0.5, 513)
h = x[1] - x[0]


# Set parameters
m = 2
c = np.zeros_like(x)
tau = 0.001
eps = 1e-3             #1.0**(-3)
M = 0.5
b = (np.sqrt(3) * M / 8)**(2 / 3) 
gamma = 1e-3
h0 = 15
t0 = 1 / gamma * (b / h0)**3
t = t0 * gamma
z = np.maximum(1 / t**(1 / 3) * (b - (1 / (12 * t**(2 / 3))) * x**2), 0)   #z = rho
nu = np.zeros_like(z)
delta = 1e-6


print(z)

#def lu_decomposition(matrix):
   
mu = 1 / (delta + m * z**(m-1))

a = mu + 2 * tau / h**2
b = c = np.full_like(mu, - tau / h**2)
nu = mu * z**m

#LU decomposition
for i in range(len(x) - 1):
   b[i] /= a[i]
   a[i+1] -= b[i] * c[i]

#forward elimination
for i in range(len(x) - 1):
    nu[i+1] -= b[i] * nu[i] 
    
#backward substitution
nu[-1] /= a[-1]

for i in range(len(x)-2,-1,-1):
    nu[i] = (nu[i] - c[i] * nu[i+1]) / a[i]
    
print(nu)
    
"""

#mu = 1 / (delta + m * z**(m-1))
a = np.array([2.,4.,-1.])
#d = a
b = np.array([4.,3.])
c = np.array([3.,-3.])
#l = np.array([0.,0.])
u = np.array([8.,3.,3.])
#y = z
x = np.zeros_like(u)


def tri_lu():
    #a = mu + 2 * tau / h**2
    #b = np.full_like(mu, - tau / h**2)
    #b = c = np.delete(b, -1)
    
    
    #LU decomposition
    for i in range(len(x) - 1):
        b[i] /= a[i]
        a[i+1] -= b[i] * c[i]

    #forward elimination
    for i in range(len(x) - 1):
        u[i+1] -= b[i] * u[i] 
        
    #backward substitution
    u[-1] /= a[-1]
    for i in range(len(x)-2,-1,-1):
        u[i] = (u[i] - c[i] * u[i+1]) / a[i]
    
    return u
print(tri_lu())


a = np.array([2.,4.,-1.])
#d = a
b = np.array([4.,3.])
c = np.array([3.,-3.])
#l = np.array([0.,0.])
u = np.array([8.,3.,3.])
#LU decomposition
#for i in range(2):
#    l[i] = b[i] / d[i]
#    d[i+1] = a[i+1] - l[i] * c[i]
for i in range(2):
    b[i] /= a[i]
    a[i+1] -= b[i] * c[i]

print(c)
print(a)
print(b)

#forward elimination
"""
for i in range(2):
    y[i+1] = z[i+1] - b[i] * y[i]
print(y)
"""
for i in range(2):
    u[i+1] -= b[i] * u[i] 
print(u)
#backward substitution
"""
x[-1] = z[-1] / a[-1]
for i in range(1, -1, -1):
    x[i] = (z[i] - c[i] * x[i+1]) / d[i]
print(x)
"""

u[-1] /= a[-1]
for i in range(1,-1,-1):
    u[i] = (u[i] - c[i] * u[i+1]) / a[i]
print(u)
    

# Example usage:
"""
D = 1
dt = 0.0002


x = np.linspace(-1,1,101)
dx = x[1] - x[0]
print (D*dt/(dx*dx))

u = np.zeros_like(x)
u[101//2] = 15.0
new_u = np.zeros_like(x)

plt.ylim([-0.1, 15.1])
for j in range(50):
    if j%5 == 0:
        plt.plot(x, u)

    u[0] = u[-1] = 0       # 境界条件。端は水平
    u[1:-1] = u[1:-1] + D*dt/(dx*dx)*(u[2:]-2*u[1:-1]+u[:-2])
    #u, new_u = new_u, u.copy()
plt.show()
"""   
"""
# パラメータ
L = 1.0  # 空間の長さ
T = 1.0  # 計算時間
Nx = 100  # 空間の離散化数
Nt = 1000  # 時間の離散化数
alpha = 0.01  # 熱伝導率

# 空間と時間の離散化
x = np.linspace(0, L, Nx)
t = np.linspace(0, T, Nt)
dx = x[1] - x[0]
dt = t[1] - t[0]

# 初期条件
u = np.zeros(Nx)
u_new = np.zeros(Nx)
u[int(0.4 * Nx):int(0.6 * Nx)] = 1.0  # 初期温度分布

# 解の保存用配列
u_history = []

# 熱方程式の数値解法
for n in range(Nt):
    u_new[1:-1] = u[1:-1] + alpha * (u[:-2] - 2 * u[1:-1] + u[2:])
    
    # 境界条件
    u_new[0] = 0
    u_new[-1] = 0
    
    u, u_new = u_new, u.copy()  # 次のステップのために更新
    
    u_history.append(u.copy())  # 解の履歴を保存

# 結果のプロット
for i, u in enumerate(u_history):
    if i % 10 == 0:  # 10ステップごとにプロット
        plt.plot(x, u, label=f"t = {i * dt:.2f}")

plt.xlabel("空間 (x)")
plt.ylabel("温度 (u)")
plt.legend()
plt.show()
"""