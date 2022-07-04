import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import interpolate
from matplotlib import cm
import matplotlib.collections as mcoll
import matplotlib.path as mpath
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection

def plot_trajectory(ax, x, y, z=None, colormap = 'jet', num_of_points = None, linewidth = 1, k = 3, plot_waypoints=False, markersize = 0.5, alpha=1, zorder = 20):

	if z is None:

		tck,u = interpolate.splprep([x,y],s=0.0, k=k)
		x_i,y_i = interpolate.splev(np.linspace(0,1,num_of_points),tck)
		points = np.array([x_i,y_i]).T.reshape(-1,1,2)
		segments = np.concatenate([points[:-2], points[1:-1], points[2:]], axis=1)
		lc = LineCollection(segments, norm = plt.Normalize(0, 1),cmap=plt.get_cmap(colormap), linewidth=linewidth,alpha=alpha, zorder = zorder)
		lc.set_array(np.linspace(0,1,len(x_i)))
		ax.add_collection(lc)
		if plot_waypoints:
			ax.scatter(x,y, s=10,  cmap = plt.get_cmap(colormap)(len(x)), norm = plt.Normalize(0, 1), marker = 'o', zorder = zorder+1)
	else:
		tck,u = interpolate.splprep([x,y,z],s=0.0)
		x_i,y_i,z_i= interpolate.splev(np.linspace(0,1,num_of_points), tck)
		points = np.array([x_i,y_i,z_i]).T.reshape(-1,1,3)
		segments = np.concatenate([points[:-2], points[1:-1], points[2:]], axis=1)
		lc = Line3DCollection(segments, norm = plt.Normalize(0, 1),cmap=plt.get_cmap(colormap), linewidth=linewidth,alpha=alpha)
		lc.set_array(np.linspace(0,1,len(x_i)))
		ax.add_collection(lc)
		ax.scatter(x,y,z,'k')
		if plot_waypoints:
			ax.plot(x,y,'kx')

	ax.plot()

def interpolate_path(path):

	tck, u = interpolate.splprep(path.T, s=3.0)
	x_i, y_i = interpolate.splev(np.linspace(0, 1, 400), tck)

	return np.column_stack((x_i, y_i))

trajectories = pd.read_csv('/Users/samuel/MultiAgentPatrollingProblem/Evaluation/DRLNetworked_paths.csv')
trajectories = trajectories[trajectories['Run'] == 0].sort_values('Step')
navigation_map = np.genfromtxt('../Environment/example_map.csv', delimiter=',')
x_centroid = trajectories.groupby('Step')['x'].mean().to_numpy()
y_centroid = trajectories.groupby('Step')['y'].mean().to_numpy()


with plt.style.context('seaborn'):

	fig, axs = plt.subplots(1, 4)
	navigation_map[np.where(navigation_map == 0)] = np.nan

	cmaps = ['Reds_r', 'Greens_r', 'Purples_r', 'Blues_r']
	colors = ['red', 'green', 'blue', 'orange']


	for indx, ax in enumerate(axs):

		ax.imshow(navigation_map, cmap='gray', interpolation='nearest', zorder=10)

		ax.set_yticklabels([])
		ax.set_xticklabels([])

		for veh in pd.unique(trajectories['vehicle']):

			veh_path = trajectories[trajectories['vehicle'] == veh][['x', 'y']].to_numpy()
			try:
				plot_trajectory(ax, x=veh_path[:, 1], y=veh_path[:, 0], colormap=cmaps[veh % 4] if indx == veh else 'Greys', num_of_points=400, alpha= 1 if indx == veh else 0.15, linewidth=2, plot_waypoints=False, markersize=20)
			except:
				ax.plot(veh_path[:, 1], veh_path[:, 0], c=colors[veh % 4])


	"""
	try:
		i_path = interpolate_path(np.column_stack((x_centroid, y_centroid)))
		plot_trajectory(ax, x=i_path[:, 1], y=i_path[:, 0], colormap='Reds', num_of_points=400)
	except:
		ax.plot(y_centroid, x_centroid, c='Green')
	"""

	plt.show()


