import numpy as np
import matplotlib.pyplot as plt
#"""
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