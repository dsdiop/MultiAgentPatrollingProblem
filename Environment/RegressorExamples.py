import sys
sys.path.append('.')
import matplotlib.pyplot as plt
import numpy as np
from groundtruthgenerator import GroundTruth
from sklearn.neighbors import KNeighborsRegressor, RadiusNeighborsRegressor
from sklearn.kernel_ridge import KernelRidge
#from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel as W
from sklearn.linear_model import HuberRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from GPytorchModels import GaussianProcessRegressor
import time




N = 50
navigation_map = np.ones((N, N))
X = np.column_stack(np.where(navigation_map == 1))

gt = GroundTruth(navigation_map, max_number_of_peaks=6, is_bounded = True, seed = 0)
z = gt.read()

# Create the regressor and sample 5 new samples #
# regressor = KNeighborsRegressor(n_neighbors=1, weights='distance')
# regressor = RadiusNeighborsRegressor(radius=10, weights='distance')
# regressor = KernelRidge(kernel='rbf', gamma=5e-2, alpha=1e-3)
# regressor = GaussianProcessRegressor(kernel=RBF(5.0, length_scale_bounds=(0.01, 10.0)) + W(0.001), n_restarts_optimizer=10)
# regressor = SVR(kernel='rbf', C=1e5, gamma=0.1)
# regressor = DecisionTreeRegressor(criterion='absolute_error', max_depth=50, min_samples_leaf=1, max_leaf_nodes=10)
regressor = GaussianProcessRegressor(lengthscale_bounds = (1.0, 10), initial_lengthscale=5.0, noise=1e-4, lr = 0.1, n_iterations = 20, device = 'cpu')

X_sample = X[np.random.randint(0,len(X), 5)]
Y_sample = z[X_sample[:,0], X_sample[:,1]]
regressor.fit(X_sample, Y_sample)

z_predicted = regressor.predict(X)
z_predicted = z_predicted.reshape((N,N))

fig, axs = plt.subplots(1,3)
d0 = axs[0].imshow(z, vmin=0.0, vmax=1.0)
d1 = axs[1].imshow(z_predicted, vmin=0.0, vmax=1.0)
d3, = plt.plot(X_sample[:,1], X_sample[:,0], 'xr')
d4 = axs[2].imshow(100*np.abs((z_predicted - z)/z), vmin=0, vmax=100)



def onclick(event):

    global X_sample, Y_sample, regressor, z_predicted

    t0 = time.time()

    new_point = np.array([event.ydata, event.xdata]).astype(int)

    new_meas = z[new_point[0], new_point[1]]

    z_prev = z_predicted.copy()

    X_sample = np.row_stack((X_sample, new_point))
    Y_sample = np.append(Y_sample, [new_meas])


    regressor.fist(X_sample, Y_sample, verbose=True, optimize= len(X_sample) < 20)

    z_predicted = regressor.predict(X)
    z_predicted = z_predicted.reshape((N,N))

    print("time: ", time.time()-t0)


    d1.set_data(z_predicted)
    d3.set_data(X_sample[:,1], X_sample[:,0])
    d4.set_data(100*np.abs((z_predicted - z)/z))

    fig.canvas.draw()
    fig.canvas.flush_events()



cid = fig.canvas.mpl_connect('button_press_event', onclick)

plt.ion()
plt.show(block=True)









