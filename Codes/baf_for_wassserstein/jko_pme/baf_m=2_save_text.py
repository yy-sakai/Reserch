import numpy as np
import matplotlib.pyplot as plt
import os
import time
from numba import njit

image_root = "images/baf_m=2/"
os.makedirs(image_root, exist_ok=True)

image_save = "images/baf_tau/"
os.makedirs(image_save, exist_ok=True)


# Wasserstein distance \int \phi d\nu + \int \phi^c d\mu
def w2(phi, psi, mu, nu):
    return np.sum(phi * nu + psi * mu)


@njit
def convex_hull(x, y):
    # 関数f(x, y)のconvex hullを計算する。(y_i = f(x_i))
    # x座標とy座標を入力すると、convex hullのx座標の添字i(x_i)と傾き（その点と一つ前の点で作る傾き）をreturnする。

    l = [(0, -np.inf)]  # np.inf = ∞,  l = [(添字（一番左の点）(x_0)、傾き-∞)]

    for i, (nx, ny) in enumerate(zip(x[1:], y[1:])):  # nx:x_i(x座標)Now x, ny:y_i(y座標), zip:(リストやタプルなど）の要素を同時に取得（x,yを同時に取得）
        # 凸関数の場合、2点でできる傾きは単調増加
        while True:
            pi, pv = l[-1]  # pi:一つ前の(Previous)「点の添字番号（左から数えた点の番号(x_i)）」pv: 一つ前の「現在の点と前の点で作る直線の傾きv」
            v = (ny - y[pi]) / (nx - x[pi])  # v:現在の点と一つ前の点で作った直線の傾き
            # print('v = ', v)
            if v <= pv:  # 傾きが一つ前の傾きより小さかったら
                del l[-1]  # 一つ前の点（添字、傾き）を消す
                # print(l)
            else:  # 傾きがひとつ前の傾きより大きい
                l.append((i + 1, v))  # 点の情報（添字、傾き）をlに追加する
                # print(l)
                break

    return [j for j, _ in l], [v for _, v in l[1:]]  # lは一つの組に2つ要素。j, _という名前でとってくるが、jのみ返す。vも同様.


@njit
def legendre_fenchel(x, y, p):
    # y = f(x)で与えられるinput(x, y)に対し、Legendre-Fenchel transformを計算する
    # (f^*(p)=sup_x(px - f(x)) = inf_x(f(x)-px), 傾きvを固定し,x[i] p - y[i]が最大となるx[i]を探す。すなわち、傾きvがある点x[i]での∇f(x[i])となるx[i]を探す

    # print(p)                                      #x = p: x座標を分割した添字
    chi, v = convex_hull(x, y)  # chi:convex_hullになったx,y座標の添字(conv) v:現在と次の点を結んだ線の傾き(現在の点での接戦の傾き（微分）)
    # print(chi, v)
    v.append(np.Inf)

    t = []
    iopt = np.zeros_like(p, dtype=np.int64)  # x = v = np.linspace(-10, 10, 100),  np.zeros_like():要素全て0に初期化
    # iopt = np.zeros(len(p), dtype=np.int64)
    i = 0
    for j, p in enumerate(p):
        while (p > v[i]):  # v[i]:現在と次の点を結んだ線の傾きは単調増加より、p:[-10,10]を99等分(-10,…,10)したものに対し、v[i]^を大きくしていき、
            # print(v, s[i])              #傾きpがv[i](分割を小さく取れば∇f(x)と等しい)と等しいもしくは大きくなる時までvを大きくする
            i += 1
        iopt[j] = chi[i]
        t.append(x[chi[i]] * p - y[chi[i]])  # p=v[i]=∇f(x)となるx座標=ipot[j]とf^*(v)=tを保存

    return t, iopt


@njit
def c_transform(x, phi, p):
    """costが c(x,p) = |x - p|^2 / 2 である時の、φ phi = φ(y)の c-transform "ψ(psi) = φ^c(p) = |p|^2/2 - φ(p)" を計算する.
    c-transform の値と、 legendre fenchel transform で sup(max) となる時の 座標の添字を return する.

     φ^c(p) = inf_x( c(x, p) -  φ(x))              y = f(x) = φ(x)
            = inf_x( |x - p|^2 / 2 -  φ(x))
            = |p|^2 / 2 - sup_x( xp - (|x|^2 / 2 - φ(x)) )
            = |p|^2 / 2 - sup_x( xp - ψ(x) )        ψ(x) = |x|^2/2 - φ(x)       ①
            = |p|^2 / 2 - f^*(p)                                            ②
    """
    # p = x
    psi = 0.5 * x * x - phi  # ①
    t, index = legendre_fenchel(x, psi, p)  # t = f^*(p)       ②

    return 0.5 * p * p - np.array(t), index  # phi^c(p) = |p|^2 / 2 - f^*(p)


@njit
def push_forward2(mu, phi_c, h):
    assert mu.shape == phi_c.shape

    t_mu = np.zeros_like(mu)

    n = phi_c.shape[0]
    l_dphi_c = []
    l_det = []

    # iterate over each cell
    for i in range(n):
        # 3 neighboring cells
        um = phi_c[max(i - 1, 0)]
        u = phi_c[i]
        up = phi_c[min(i + 1, n - 1)]

        # \nabla\phi_c
        dphi_c = (up - um) / (2.0 * h)
        # print(dphi_c)
        l_dphi_c.append(dphi_c)
        # det (I - D^2\phi_c)
        det = 1.0 - (up - 2.0 * u + um) / (h * h)
        # print(det)
        l_det.append(det)
        # x - \nabla\phi_c with respect to the cell grid
        xcell = i - (dphi_c / h)

        # indices of the nearest cells
        ti = int(min(max(np.floor(xcell), 0.0), n - 1))
        tio = min(ti + 1, n - 1)

        # interpolate the density value
        mu_inter = mu[ti] * (1.0 - (xcell - np.floor(xcell))) + mu[tio] * (xcell - np.floor(xcell))

        t_mu[i] = mu_inter * det

    return t_mu


def lap_solve_modified(f, theta_1, theta_2):
    """Solves (\theta1 I - \theta2 \Delta)u = f with Neumann boundary condition on [0,1].
    f needs to be given at all nodes including the endpoints. The mean of f is set to zero.
    """
    # even periodic extension to get cosine series; imaginary part of the result will be zero
    pf = np.concatenate((f, f[-2:0:-1]))
    ff = np.fft.rfft(pf)
    xi = np.linspace(0, 1, len(f))
    N = len(f) - 1
    ff /= (theta_2 * 4 * np.sin(0.5 * np.pi * xi) ** 2 * N**2) + theta_1
    # perform inverse fft and remove the even periodic extension
    return np.fft.irfft(ff)[: len(f)]


@njit
def gauss(f, theta_1, theta_2):
    a = np.full_like(x, -theta_2 / h**2)
    b = np.full_like(x, theta_1 + 2 * theta_2 / h**2)
    c = np.full_like(x, -theta_2 / h**2)
    c[0] = c[0] / b[0]
    f[0] = f[0] / b[0]
    for i in range(1, len(x)):
        c[i] = c[i] / (b[i] - a[i] * c[i - 1])
        f[i] = (f[i] - a[i] * f[i - 1]) / (b[i] - a[i] * c[i - 1])
    for i in range(len(x) - 3, 0, -1):
        # for i in reversed(range(1, len(x)-2)):
        f[i] = f[i] - c[i] * f[i + 1]

    return f


# ascent step of J(phi) = \int phi dnu + \int phi^c dmu
# fills phi and phi_c, returns new sigma
# using Jacobian push forward (push_forward2)
# common ascent scheme


def ascent(phi, phi_c, mu, nu):
    phi_c, _ = c_transform(x, tau * phi, x)  # 1-1  phi_c, _ = c_transform(x, phi, p)
    phi_c /= tau

    nmu = np.max(np.abs(mu))
    theta_1 = 1 / (2 * gamma)
    theta_2 = tau * nmu

    pfwd = push_forward2(mu, tau * phi, h)  # 1-2-1     pfwd : T_{\phi\#}\mu = \mu(x - \tau \nabla \phi(x))|det(I - \tau D^2\phi_c))|
    rho = nu - pfwd  # 1-2-2     rho = \nu - T_{\phi\#}\mu　＝ \delta U^*(- \phi) - T_{\phi\#}\mu
    # TODO: This is by far the slowest part of the algorithm

    # In one dimension, Gaussian elimination is faster than the Fast Fourier Transform.
    # lp = lap_solve_modified(rho, theta_1, theta_2)                             # 1-2-3     lp: \nabla_{\dot{H}^1} J(\phi_n) = (- \Delta)^{-1} * rho
    lp = gauss(rho, theta_1, theta_2)
    phi += lp  # 1-2-4   phi_{n + 1/2} = phi_n + sigma * lp
    #####################################################################
    phi_c, _ = c_transform(x, tau * phi, x)  # 2    psi_{n + 1/2} = (phi_{n + 1/2})^c
    phi_c /= tau
    H1_sq = np.mean(rho * lp)  #######       ?
    return H1_sq, phi, phi_c, pfwd  #  ?


def baf(tau):  # JKO scheme
    # Initialisation of parameters
    t = t0 * gamma
    nu = np.maximum(1 / t ** (1 / 3) * (b - (1 / (12 * t ** (2 / 3))) * x**2), 0)  # nu = rho
    mu = nu
    # U(\rho) = 1 / m-1 \int \rho^m dx
    phi = -gamma * (m / (m - 1)) * nu ** (m - 1)  # \phi_0 = \phi^(0) = -\delta U(\nu^(0)) =  \delta U(\rho^(0))
    psi = np.zeros_like(nu)
    timestep = np.arange(0, 2, tau)
    # stepsize = timestep[1] - timestep[0] = tau

    error = 0

    start = time.process_time()
    for real_t in timestep:
        diff = 1
        count = 0
        # The back-and-forth scheme for solving J(phi) and I(psi)
        while diff >= eps:
            if count > 100:
                break

            nu = (((m - 1) / (m * gamma)) * np.maximum(c - phi, 0)) ** (1 / (m - 1))  # \rho_*(x) = \delta U^*(- \phi)
            H1_sq, phi, psi, pfwd = ascent(phi, psi, mu, nu)  # phi = phi_{k + 1/2}, psi = psi_{k + 1/2}

            # Calculate residual $||\nabla U^*(- \varphi) - T_{\phi \#} \mu||_{L^1(\Omega)}
            # L1 norm
            diff = sum(abs(nu - pfwd) * h)
            # print(f'{real_t + tau:.4}:(H¹)² = {H1_sq:.3}, diff = {diff:.5}, baf_roop = {count}')

            ##################################################################################

            psi_c, _ = c_transform(x, tau * psi, x)  # 3-1  phi_c, _ = c_transform(x, phi, p)
            psi_c /= tau

            nu = (((m - 1) / (m * gamma)) * np.maximum(c - psi_c, 0)) ** (1 / (m - 1))  # nu = T_{\psi \#} \delta U^* (- \psi^c)
            H1_sq, psi, phi, pfwd = ascent(psi, phi, nu, mu)

            count += 1

        mu = (((m - 1) / (m * gamma)) * np.maximum(c - phi, 0)) ** (1 / (m - 1))

        t = (real_t + tau + t0) * gamma
        u = np.maximum(1 / t ** (1 / 3) * (b - (1 / (12 * t ** (2 / 3))) * x**2), 0)
        area = sum((u[1:] + u[:-1]) * h / 2)  # trapezoidal formula
        error += (tau / 2) * sum(abs((u - nu)[1:] + (u - nu)[:-1]) * h / 2)  # error = (2 / \tau) * \sigma_{n=0}^{2 / \tau} \int |\rho(n*\tau + t0, x) - \rho^(n)(x)| dx

    # error /= (2 / tau)  # error = (2 / \tau) * \sigma_{n=0}^{2 / \tau} \int |\rho(n*\tau + t0, x) - \rho^(n)(x)| dx
    #error = f"{round(error, 7):.2e}"
    realtime = f"{(time.process_time() - start):.3g}"

    return error, realtime, area


class Hist:
    pass


hist = Hist()

hist.tau = []
hist.N_tau = []
hist.error = []

# whether to plot all timesteps and save the time step data
track = True
H1_sq = 0

# Set parameters
x = np.linspace(-0.5, 0.5, 4001)
h = x[1] - x[0]
m = 2
c = np.zeros_like(x)
tau = 0.4
eps = 1e-6  # 1.0**(-3)
M = 0.5
b = (np.sqrt(3) * M / 8) ** (2 / 3)
gamma = 1e-3
h0 = 15
t0 = 1 / gamma * (b / h0) ** 3


# Load the respective function first.
print("tau = ", tau)
start = time.process_time()
error, realtime, area = baf(tau)
print("error = ", error)
print(f"Elapsed {realtime}s")


# Save the ERROR and TIME in a text file.
with open("Codes/result/result_baf_gauss4000_eps=1e-6.tex", "w") as f:
    f.write("\\begin{tabular}{llll} \n")
    f.write("\hline \n")
    f.write("$\\tau$  & $N_\\tau$  &  Error & Times$(s)$  \\\ \n")
    f.write("\hline \hline \n")

    # Repeat the operation of halving the value of tau 7 times.
    for i in range(11):
        print("tau = ", tau)
        error, realtime, area = baf(tau)
        print("error = ", error)
        print(f"Elapsed {realtime}s")
        print("")
        # String to be saved in a text file.
        f.write(f"{tau}  & {int(2 / tau)} & \\num{{{error}}} & {realtime} \\\ \n")

        hist.tau.append(tau)
        hist.N_tau.append(int(2 / tau))
        hist.error.append(error)
        tau /= 2
        # if i == 6:
        #     tau = 0.0001

    f.write("\hline \n")
    f.write("\end{tabular} \n")


print(hist.N_tau)
print(hist.error)

# plt.plot(hist.N_tau[::-1], hist.error[::-1])
# plt.show()


# Plot the expected scale of error for each tau.
plt.loglog(hist.N_tau, hist.error, marker=".", label=r"error")
plt.loglog(hist.N_tau, np.ones_like(hist.tau) / hist.N_tau,marker=".",label=r"$\frac{1}{N_{\tau}}$")
plt.loglog(hist.N_tau, np.ones_like(hist.tau) / np.square(hist.N_tau), marker='.', label=r'$\frac{1}{N_{\tau}^2}$')
# plt.semilogy(hist.N_tau, np.ones_like(hist.tau) / np.power(hist.N_tau, 3), marker='.', label=r'$\frac{1}{N_{\tau}^3}$')
#plt.loglog(hist.N_tau, hist.error, marker=".", label=r"error")
plt.xlabel(r"log($N_{\tau}$)")
plt.ylabel("log(error)")
plt.legend()
plt.savefig(f"{image_save}log_error.png")
plt.close()

# Make sure they are parallel.

plt.loglog(hist.N_tau, abs(np.array(hist.error, dtype=float) / (np.ones_like(hist.tau) / hist.N_tau)),marker=".",label=r"$error / \frac{1}{N_\tau}$")
# plt.loglog(hist.N_tau,np.array(hist.error, dtype=float) - (np.ones_like(hist.tau) / np.square(hist.N_tau)), marker='.', label=r'$error - \frac{1}{N_{\tau}^2}$')
plt.legend()
plt.show()
