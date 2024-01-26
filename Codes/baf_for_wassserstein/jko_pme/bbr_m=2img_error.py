import numpy as np
import matplotlib.pyplot as plt

image_root = "images/bbr_tau/"

img_tau000625 = np.load(f"{image_root}error_tau=0.00625.npy")
# img_tau00125 = np.load(f'{image_root}tau=0.0125.npy')
img_tau0025 = np.load(f"{image_root}error_tau=0.025.npy")
# img_tau005 = np.load(f'{image_root}tau=0.05.npy')
img_tau01 = np.load(f"{image_root}error_tau=0.1.npy")
# img_tau02 = np.load(f'{image_root}tau=0.2.npy')
img_tau04 = np.load(f"{image_root}error_tau=0.4.npy")



t = [0, 0.4, 0.8, 1.2, 1.6, 2.0]
plt.plot(t, img_tau04, label=r"$\tau = 0.4$")

t = np.arange(0, 2.1, 0.1)
plt.plot(t, img_tau01, label=r"$\tau = 0.1$")
plt.plot(t, img_tau0025, label=r"$\tau = 0.025$")
plt.plot(t, img_tau000625, label=r"$\tau = 0.00625$")
plt.xlabel("t")
plt.ylabel("error = computed - exact")
plt.legend()
plt.savefig(f"{image_root}error_bbr.png")
plt.close()