import numpy as np
import matplotlib.pyplot as plt

image_root = "../images/euler_tau/"

img_tau001 = np.load(f'{image_root}tau=0.001.npy')
img_tau0001 = np.load(f'{image_root}tau=0.0001.npy')
img_tau00005 = np.load(f'{image_root}tau=5e-05.npy')
img_tau000025 = np.load(f'{image_root}tau=2.5e-05.npy')
img_exact = np.load(f'{image_root}exact.npy')

x = np.linspace(-0.5, 0.5, 513)
h = x[1] - x[0]
t = [0, 0.4, 0.8, 2]

plt.ylim([-0.1, 15.1])
for i in range(4):
    plt.ylim([-0.1, 15.1])
    plt.plot(x, img_exact[i], "--", label=r'$exact$')
    plt.plot(x, img_tau001[i],label=r'$\tau = 0.001$')
    plt.plot(x, img_tau0001[i],label=r'$\tau = 0.0001$')
    plt.plot(x, img_tau00005[i],label=r'$\tau = 0.00005$')
    plt.plot(x, img_tau000025[i],label=r'$\tau = 0.000025$')
    plt.legend()
    plt.savefig(f'{image_root}t={t[i]}.png')
    plt.close()
    
print(f'Plots saved in {image_root}')