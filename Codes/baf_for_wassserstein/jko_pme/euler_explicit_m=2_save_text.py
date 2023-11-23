import numpy as np
import matplotlib.pyplot as plt
import os
import time

image_root = "../images/euler/"
os.makedirs(image_root, exist_ok = True)
image_save = "../images/euler_tau/"
os.makedirs(image_save, exist_ok = True)
# Wasserstein distance \int \phi d\nu + \int \phi^c d\mu
def w2(phi, psi, mu, nu):
    return np.sum(phi * nu + psi * mu)

# ascent step of J(phi) = \int phi dnu + \int phi^c dmu
# fills phi and phi_c, returns new sigma
# using Jacobian push fofward (push_forward2) 
# common ascent scheme
def euler(tau): # JKO scheme
    # Initialisation of parameters
    t = t0 * gamma
    nu = np.maximum(1 / t**(1 / 3) * (b - (1 / (12 * t**(2 / 3))) * x**2), 0)   #nu = rho
    timestep = np.arange(0, 2, tau)
    #stepsize = timestep[1] - timestep[0] = tau
    
    error = 0
    start = time.process_time()
    # JKO scheme
    for real_t in timestep:
        nu[0] = nu[-1] = 0
        nu[1:-1] = nu[1:-1] + ((tau * gamma)/ h**2) * (nu[:-2]**m -2*nu[1:-1]**m + nu[2:]**m)
        t = (real_t + tau + t0) * gamma
        u = np.maximum(1 / t**(1 / 3) * (b - (1 / (12 * t**(2 / 3))) * x**2), 0)
        area = sum((u[1:] + u[:-1]) * h / 2)   #trapezoidal formula 
        error += sum(abs((u - nu)[1:] + (u - nu)[:-1]) * h / 2)  #error = (2 / \tau) * \sigma_{n=0}^{2 / \tau} \int |\rho(n*\tau + t0, x) - \rho^(n)(x)| dx

    error /= 2 / tau  #error = (2 / \tau) * \sigma_{n=0}^{2 / \tau} \int |\rho(n*\tau + t0, x) - \rho^(n)(x)| dx
    error = f'{round(error, 7):.3e}'
    realtime = f'{(time.process_time() - start):.3e}'
    
    return error, realtime, area

# whether to plot all timesteps and save the time step data
track = True
H1_sq = 0

# Set parameters
x = np.linspace(-0.5, 0.5, 513)
h = x[1] - x[0]
m = 2
c = np.zeros_like(x)
tau = 0.0001
# tau <= 1 / (2 * m * gamma) * h**2 = 0.00095367431640625 Stability conditions(maybe)
# but 0.0002 is not stable

eps = 1e-3             #1.0**(-3)
M = 0.5
b = (np.sqrt(3) * M / 8) ** (2 / 3)
gamma = 1e-3
h0 = 15
t0 = 1 / gamma * (b / h0) ** 3
#u = np.zeros_like(x)

# Load the respective function first.
print('tau = ', tau)
start = time.process_time()
error, realtime, area = euler(tau)
print('error = ', error)
print(f'Elapsed {realtime}s')
print('')


# Save the ERROR and TIME in a text file.
with open('result_euler.tex', 'w') as f:
    f.write('\\begin{tabular}{llll} \n')
    f.write('\hline \n')
    f.write('$\\tau$  & $N_\\tau$  &  Error & Times$(s)$  \\\ \n')
    f.write('\hline \hline \n')
    
    
    # Repeat the operation of halving the value of tau 7 times.
    for _ in range(3):
        print('tau = ', tau)
        error, realtime, area = euler(tau)
        print('error = ', error)
        print(f'Elapsed {realtime}s')
        print('')
        #String to be saved in a text file.
        f.write(f'{tau}  & {int(2 / tau)} & \\num{{{error}}} & \\num{{{realtime}}} \\\ \n') 

        tau /= 2

    f.write('\hline \n')
    f.write('\end{tabular} \n')