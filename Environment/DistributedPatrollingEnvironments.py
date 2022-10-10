import gym
import numpy as np
import matplotlib.pyplot as plt


class DistributedVehicle:
	default_config_dict = {
		"navigation_map": np.ones((50, 50)),
		"distance_budget": 100,
		"radius": 3,
		"forget_factor": 0.01,
		"ground_truth": np.random.rand((50, 50)),
		"initial_position": np.array([10, 20]),
		"movement_length": 3,

	}

	def __init__(self, agent_id, default_config: dict):
		# Unpack config values #
		self.navigation_map = default_config["navigation_map"]
		self.distance_budget = default_config["distance_budget"]
		self.detection_radius = default_config["radius"]
		self.forget_factor = default_config["forget_factor"]
		self.ground_truth_field = default_config["ground_truth"]
		self.initial_position = default_config["initial_position"]
		self.movement_length = default_config["movement_length"]
		self.agent_id = agent_id

		# Create the matrices #
		self.position = self.initial_position.copy()
		self.information_matrix = np.zeros_like(self.navigation_map)
		self.idleness_matrix = np.ones_like(self.navigation_map)
		self.precision_matrix = np.zeros_like(self.navigation_map)
		self.detection_mask = self.compute_detection_mask()
		self.fleet_positional_map = np.zeros_like(self.navigation_map)

		# Vahicle values #
		self.distance = 0
		self.number_of_collisions = 0
		self.state = None
		self.waypoints = None
		self.d = None

	def reset(self,
	          initial_position: np.ndarray,
	          new_ground_truth_field: np.ndarray):
		""" Reset the state of the matrixes and the vehicle.
         Update the state taking in account the other vehicles model """

		# Reset the ground_truth #
		self.ground_truth_field = new_ground_truth_field
		# Reset the position
		self.position = initial_position
		self.waypoints = np.atleast_2d(self.position)
		# Reset the matrixes
		self.information_matrix = np.zeros_like(self.navigation_map)
		self.idleness_matrix = np.ones_like(self.navigation_map)
		self.precision_matrix = np.zeros_like(self.navigation_map)
		self.detection_mask = self.compute_detection_mask()
		self.fleet_positional_map = np.zeros_like(self.navigation_map)

		self.distance = 0
		self.number_of_collisions = 0

		# Update the model
		self.update_model()

		# Generate state #
		self.state = self.process_state()

		return self.state

	def compute_detection_mask(self):

		known_mask = np.zeros_like(self.navigation_map)

		px, py = self.position.astype(int)

		# State - coverage area #
		x = np.arange(0, self.navigation_map.shape[0])
		y = np.arange(0, self.navigation_map.shape[1])

		# Compute the circular mask (area) of the state 3 #
		mask = (x[np.newaxis, :] - px) ** 2 + (y[:, np.newaxis] - py) ** 2 <= self.detection_radius ** 2

		known_mask[mask.T] = 1.0

		return known_mask

	def update_model(self, other_agents_info):
		""" Update the model using only the information from the self vehicle """

		# Update the detection mask #
		self.detection_mask = self.compute_detection_mask()

		# Update the precision map [P() OR Mask]#
		self.precision_matrix = np.logical_or(self.detection_mask, self.precision_matrix) * self.navigation_map

		# Update the self information map #
		self.information_matrix = self.ground_truth_field * self.precision_matrix

		# Update the idleness matrix
		self.idleness_matrix += self.forget_factor
		self.idleness_matrix = self.idleness_matrix - self.detection_mask
		self.idleness_matrix = np.clip(self.idleness_matrix, 0.0, 1.0) * self.navigation_map

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
			self.position = next_position
			self.waypoints = np.vstack((self.waypoints, [self.position]))
			self.distance += np.linalg.norm(self.position - next_position)

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

	def process_state(self):

		state = np.zeros((5, *self.navigation_map.shape))

		# State 0 ->  Obstacles map #
		state[0] = self.navigation_map
		# State 1 -> Temporal mask
		state[1] = self.idleness_matrix
		# State 2 -> Known information
		state[2] = self.information_matrix
		# State 3 -> Self position
		state[3][int(self.position[0]), int(self.position[1])] = 1.0
		# State 4 -> Other positions
		state[4] = self.fleet_positional_map

		return state

	def render(self):

		if self.d is None:
			self.d = []
			self.fig, self.axs = plt.subplots(1, 5)
			for i, ax in enumerate(self.axs):
				self.d.append(ax.imshow(self.state[i]))
		else:
			for i, d in enumerate(self.d):
				d.set_data(self.state[i])

		self.fig.canvas.draw()
		plt.draw()
		plt.pause(0.1)


class DistributedFleet:
	default_config_dict = {
		"vehicle_config": DistributedVehicle.default_config_dict,
		"navigation_map": np.ones((50, 50)),
		"random_initial_positions": False,
		"initial_positions": np.zeros((1, 2)),
		"number_of_agents": 1,
		"ground_truth": None
	}

	def __init__(self, default_config):

		agent_config = default_config["vehicle_config"]
		agent_config["navigation_map"] = default_config["navigation_map"]

		self.number_of_agents = default_config["number_of_agents"]
		self.initial_positions = default_config["number_of_agents"]
		self.random_initial_positions = default_config["random_initial_positions"]
		self.navigation_map = default_config["navigation_map"]
		self.ground_truth = default_config["ground_truth"]
		self.agents = [DistributedVehicle(agent_id=i, default_config=agent_config) for i in range(self.number_of_agents)]
		self.valid_positions = np.column_stack(np.where(self.navigation_map == 1))

	def reset(self):

		# Reset the ground truth #
		self.ground_truth.reset()

		# Reset every agent #
		if self.random_initial_positions:
			new_positions = self.valid_positions[np.random.choice(np.arange(0, len(self.valid_positions), self.number_of_agents, replace=False))]
		else:
			new_positions = self.initial_positions.copy()

		for i, agent in enumerate(self.agents):
			agent.reset(new_positions[i], self.ground_truth.read())

		# Now, for t


if __name__ == '__main__':
	nav_map = np.genfromtxt('./example_map.csv', delimiter=',')
