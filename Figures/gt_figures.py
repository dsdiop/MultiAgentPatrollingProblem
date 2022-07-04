import numpy as np
import matplotlib.pyplot as plt
from Environment.groundtruthgenerator import GroundTruth
from mpl_toolkits.axes_grid1 import make_axes_locatable

nav_map = np.genfromtxt('../Environment/example_map.csv', delimiter=',')
X, Y = np.meshgrid(np.arange(nav_map.shape[0]), np.arange(nav_map.shape[1]))

mask = np.full_like(nav_map, np.nan)
mask[np.where(nav_map == 0)] = 1
gt = GroundTruth(1-nav_map, resolution=1)

N = 1
with plt.style.context('seaborn'):
	fig, axs = plt.subplots(1,1)

	axs = [axs] if not hasattr(axs, '__iter__') else axs


	# Z = None
	Z = np.genfromtxt('../Evaluation/evaluation_map.csv')
	for ax in axs:

		if Z is None:
			Z = gt.read()

		im = ax.imshow(Z, cmap='jet', interpolation='bicubic')
		ax.imshow(mask, cmap='gray_r', zorder=10)
		ax.set_xticks([])
		ax.set_yticks([])
		Zt = Z.copy()
		Zt[np.where(nav_map == 0)] = np.nan
		contours = ax.contour(Y,X,Zt.T, 3, colors='black')
		ax.clabel(contours, inline=True, fontsize=6)

		gt.reset()


	divider = make_axes_locatable(axs[-1])
	cax = divider.append_axes('right', size='5%', pad=0.05)
	cbar = fig.colorbar(im, cax=cax, orientation='vertical')
	cbar.ax.set_ylabel(r'Contamination index - $\mathcal{I}$')
	plt.tight_layout()
	plt.show()