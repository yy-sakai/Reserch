import numpy as np
import matplotlib.pyplot as plt



baf = np.load(f'../images/baf_tau/error_tau=0.00625.npy')
bbr = np.load(f'../images/bbr_tau/error_tau=0.00625.npy')

x = np.linspace(-0.5, 0.5, 513)
center_x = np.array((x[1:] + x[:-1]) / 2)

plt.plot(center_x, baf, label=r'$error_ baf$')
plt.plot(center_x, bbr, label=r'$error_ bbr$')
#plt.ticklabel_format(axis='y', style='plain')
plt.xlabel("x")
plt.ylabel("error = exact - computed")
#plt.ylim(0, 1e-4)

plt.legend()
plt.savefig(f'../images/error_tau = 0.00625.png')
plt.close()
    
print(f'Plots saved in images')