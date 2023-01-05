import sys
sys.path.append('.')

import gym
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
from GPEstimation.GaussianProcessEstimators import GPExactRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel as W, Matern
import copy

class DistributedVehicle:

	def __init__(self, agent_id, default_config: dict):

		# Unpack config values #
		self.navigation_map = default_config["navigation_map"]
		self.kernel_lengthscale = default_config["kernel_lengthscale"]
		self.initial_position = default_config["initial_position"]
		self.movement_length = default_config["movement_length"]
		self.agent_id = agent_id

		# Create attributes #
		self.position = self.initial_position.copy()
		self.measured_values = None
		self.measured_locations = None

		"""
		self.regressor = GPExactRegressor(hyperparams_bounds = {"lengthscale": (1.0, 10.0), "noise": (0.001, 1.0)}, 
											lengthscale= self.kernel_lengthscale, 
											fixed_noise = None, 
											training_iters = 20, 
											lr=0.1)
		"""
		self.regressor = GaussianProcessRegressor(kernel=RBF(5.0, length_scale_bounds=(1.0, 5.0)) + W(0.01), n_restarts_optimizer=1, alpha=0.01)

		self.model_map = np.zeros_like(self.navigation_map)
		self.model_precision = np.ones_like(self.navigation_map)

		self.visitable_positions = np.column_stack(np.where(self.navigation_map == 1.0))

		# Vahicle values #
		self.distance = 0
		self.number_of_collisions = 0
		self.waypoints = None
		self.fig = None

	def update_model(self, new_position, new_value):
		""" Update the regression model with a new sample """

		if self.measured_locations is None:
			self.measured_locations = np.atleast_2d(new_position)
			self.measured_values = np.atleast_2d(new_value)
		else:
			self.measured_values = np.vstack((self.measured_values, new_value))
			self.measured_locations = np.vstack((self.measured_locations, new_position))
		
		# Avoid non-unique values #
		self.measured_locations, indx = np.unique(self.measured_locations, axis=0, return_index=True)
		self.measured_values = self.measured_values[indx]

		self.regressor.fit(self.measured_locations, self.measured_values.flatten())


		mu, unc = self.regressor.predict(self.visitable_positions, return_std=True)

		self.model_map[self.visitable_positions[:,0], self.visitable_positions[:,1]] = mu
		self.model_precision[self.visitable_positions[:,0], self.visitable_positions[:,1]] = unc**2


	def reset(self, initial_position: np.ndarray, new_ground_truth_field: np.ndarray):
		""" Reset the state of the matrixes and the vehicle.
		 Update the state taking in account the other vehicles model """

		# Reset the ground_truth #
		self.ground_truth_field = new_ground_truth_field
		# Reset the position
		self.position = initial_position
		self.waypoints = np.atleast_2d(self.position)
		# Reset the model
		self.model_map = np.zeros_like(self.navigation_map)
		self.model_precision = np.ones_like(self.navigation_map)
		self.measured_locations = None
		self.measured_values = None

		new_value = self.ground_truth_field[int(self.position[0]), int(self.position[1])]
		self.update_model(self.position.copy(), new_value)
		# Reset other variables
		self.distance = 0
		self.number_of_collisions = 0


	def move(self, action):
		""" Move a vehicle in the direction of the action. If valid is False, the action is not performed. """

		# Compute the next attempted position #
		angle = 2 * np.pi / 8.0 * action
		movement = np.array([self.movement_length * np.cos(angle), self.movement_length * np.sin(angle)])
		next_position = self.position + movement

		if self.check_collision(next_position):
			# With a collision we increase the count #
			collide = True
			self.number_of_collisions += 1
		else:
			# Without any collisions we can update the position #
			collide = False
			self.distance += np.linalg.norm(self.position - next_position)
			self.position = next_position
			self.waypoints = np.vstack((self.waypoints, [self.position]))
			
			# Update the model
			new_value = self.ground_truth_field[int(self.position[0]), int(self.position[1])]
			self.update_model(self.position.copy(), new_value)
			
		return collide

	def check_collision(self, next_position):

		outbounds_condition = next_position[0] > self.navigation_map.shape[0] or \
							  next_position[0] < 0 or \
							  next_position[1] > self.navigation_map.shape[1] or \
							  next_position[1] < 0

		if outbounds_condition:
			return True  # There is a collision
		elif self.navigation_map[int(next_position[0]), int(next_position[1])] == 0:
			return True
		else:
			return False

	def get_valid_mask(self):

		# Compute the next attempted position #
		mask = []
		for action in range(8):
			angle = 2 * np.pi / 8.0 * action
			movement = np.array([self.movement_length * np.cos(angle), self.movement_length * np.sin(angle)])
			next_position = self.position + movement
			if not self.check_collision(next_position=next_position):
				mask.append(True)
			else:
				mask.append(False)

		return np.array(mask)

	def render(self):

		if self.fig is None:

			self.fig, self.axs = plt.subplots(1, 2)
			self.d0 = self.axs[0].imshow(self.model_map, vmin = 0.0, vmax=1.0, cmap='jet')
			self.d1 = self.axs[1].imshow(self.model_precision, vmin = 0.0, vmax=1, cmap='gray')

		else:
			
			self.d0.set_data(self.model_map)
			self.d1.set_data(self.model_precision)

		self.fig.canvas.draw()
		plt.draw()
		plt.pause(0.1)



	
		



if __name__ == '__main__':

	from ShekelGroundTruth import Shekel

	nav_map = np.ones((50,50))

	gp_config = Shekel.sim_config_template
	gp_config["navigation_map"] = nav_map
	gp = Shekel(gp_config)
	CONFIG = {"navigation_map": nav_map, 
				"kernel_lengthscale": 10,
				"initial_position": np.array([25,25]), 
				"movement_length": 3.0}
	
	agent = DistributedVehicle(0, CONFIG)
	agent2 = DistributedVehicle(1, CONFIG)

	np.random.seed(42)

	agent.reset(np.array([25,25]), gp.read())
	#agent.render()

	agent2.reset(np.array([7,7]), gp.read())
	#agent2.render()

	action = np.random.randint(0,8)
	action2 = np.random.randint(0,8)

	for t in range(150):
		
		
		colision = agent.move(action)

		if colision or t % 10 == 0:
			action = np.random.randint(0,8)

		colision2 = agent2.move(action2)

		if colision2 or t % 10 == 0:
			action2 = np.random.randint(0,8)

		#agent.render()
		#agent2.render()
		I = (agent.model_map * 1.0/agent.model_precision + agent2.model_map * 1.0/agent2.model_precision)/(1.0/agent2.model_precision + 1.0/agent.model_precision)
		plt.imshow(I, cmap='jet', vmin=0, vmax=1)
		plt.pause(0.1)

	plt.show(block=True)