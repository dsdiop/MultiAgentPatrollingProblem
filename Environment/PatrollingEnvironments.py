from abc import ABC
import gym
import numpy as np
import matplotlib.pyplot as plt
from groundtruthgenerator import GroundTruth
from scipy.spatial import distance_matrix


class DiscreteVehicle:

	def __init__(self, initial_position, n_actions, movement_length, navigation_map, detection_length):
		
		""" Initial positions of the drones """
		self.initial_position = initial_position
		self.position = np.copy(initial_position)
		""" Initialize the waypoints """
		self.waypoints = np.expand_dims(np.copy(initial_position), 0)
		""" Detection radius for the contmaination vision """
		self.detection_length = detection_length
		self.navigation_map = navigation_map
		self.detection_mask = self.compute_detection_mask()

		""" Reset other variables """
		self.distance = 0.0
		self.num_of_collisions = 0
		self.action_space = gym.spaces.Discrete(n_actions)
		self.angle_set = np.linspace(0, 2 * np.pi, n_actions, endpoint=False)
		self.movement_length = movement_length

	def move(self, action, valid=True):
		""" Move a vehicle in the direction of the action. If valid is False, the action is not performed. """

		angle = self.angle_set[action]
		movement = np.round(np.array([self.movement_length * np.cos(angle), self.movement_length * np.sin(angle)])).astype(int)
		next_position = self.position + movement
		self.distance += np.linalg.norm(self.position - next_position)

		if self.check_collision(next_position) or not valid:
			collide = True
			self.num_of_collisions += 1
		else:
			collide = False
			self.position = next_position
			self.waypoints = np.vstack((self.waypoints, [self.position]))

		self.detection_mask = self.compute_detection_mask()

		return collide

	def check_collision(self, next_position):

		if self.navigation_map[int(next_position[0]), int(next_position[1])] == 0:
			return True  # There is a collision

		return False

	def compute_detection_mask(self):
		""" Compute the circular mask """

		known_mask = np.zeros_like(self.navigation_map)

		px, py = self.position.astype(int)

		# State - coverage area #
		x = np.arange(0, self.navigation_map.shape[0])
		y = np.arange(0, self.navigation_map.shape[1])

		# Compute the circular mask (area) of the state 3 #
		mask = (x[np.newaxis, :] - px) ** 2 + (y[:, np.newaxis] - py) ** 2 <= self.detection_length ** 2

		known_mask[mask.T] = 1.0

		return known_mask

	def reset(self, initial_position):
		""" Reset the agent - Position, detection mask, etc. """

		self.initial_position = initial_position
		self.position = np.copy(initial_position)
		self.waypoints = np.expand_dims(np.copy(initial_position), 0)
		self.distance = 0.0
		self.num_of_collisions = 0
		self.detection_mask = self.compute_detection_mask()

	def check_action(self, action):
		""" Return True if the action leads to a collision """

		angle = self.angle_set[action]
		movement = np.round(np.array([self.movement_length * np.cos(angle), self.movement_length * np.sin(angle)])).astype(int)
		next_position = self.position + movement

		return self.check_collision(next_position)

	def move_to_position(self, goal_position):
		""" Move to the given position """

		assert self.navigation_map[goal_position[0], goal_position[1]] == 1, "Invalid position to move"
		self.distance += np.linalg.norm(goal_position - self.position)
		""" Update the position """
		self.position = goal_position


class DiscreteFleet:

	def __init__(self,
	             number_of_vehicles,
	             n_actions,
	             fleet_initial_positions,
	             movement_length,
	             navigation_map,
	             max_connection_distance=10,
	             optimal_connection_distance=5):

		""" Coordinator of the movements of the fleet. Coordinates the common model, the distance between drones, etc. """

		self.number_of_vehicles = number_of_vehicles
		self.initial_positions = fleet_initial_positions
		self.n_actions = n_actions
		self.movement_length = movement_length

		""" Create the vehicles object array """
		self.vehicles = [DiscreteVehicle(initial_position=fleet_initial_positions[k],
		                                 n_actions=n_actions,
		                                 movement_length=movement_length,
		                                 navigation_map=navigation_map,
		                                 detection_length=movement_length) for k in range(self.number_of_vehicles)]

		self.agent_positions = np.asarray([veh.position for veh in self.vehicles])

		# Get the redundancy mask #
		self.redundancy_mask = np.sum([veh.detection_mask for veh in self.vehicles], axis=0)
		# Get the collective detection mask #
		self.collective_mask = self.redundancy_mask.astype(bool)
		self.historic_visited_mask = self.redundancy_mask.astype(bool)
		# Reset model variables 
		self.measured_values = None
		self.measured_locations = None

		# Reset fleet-communication-restriction variables #
		self.max_connection_distance = max_connection_distance
		self.isolated_mask = None
		self.fleet_collisions = 0
		self.danger_of_isolation = None
		self.distance_between_agents = None
		self.optimal_connection_distance = optimal_connection_distance
		self.number_of_disconnections = 0

	@staticmethod
	def majority(arr: np.ndarray) -> bool:
		return arr.sum() >= len(arr) // 2

	def check_fleet_collision_within(self, veh_actions):
		""" Check if there is any collision between agents """
		
		new_positions = []

		for idx, veh in enumerate(self.vehicles):

			angle = veh.angle_set[veh_actions[idx]]
			movement = np.round(np.array([veh.movement_length * np.cos(angle), veh.movement_length * np.sin(angle)])).astype(int)
			new_positions.append(list(veh.position + movement))

		_, inverse_index, counts = np.unique(np.asarray(new_positions), return_inverse=True, return_counts=True, axis=0)

		# True if repeated #
		not_collision_within = counts[inverse_index] == 1

		return not_collision_within

	def move(self, fleet_actions):

		# Check if there are collisions between vehicles #
		self_colliding_mask = self.check_fleet_collision_within(fleet_actions)
		# Process the fleet actions and move the vehicles #
		collision_array = [self.vehicles[k].move(fleet_actions[k], valid=valid) for k, valid in zip(range(self.number_of_vehicles), self_colliding_mask)]
		# Update vector with agent positions #
		self.agent_positions = np.asarray([veh.position for veh in self.vehicles])
		# Sum up the collisions for termination #
		self.fleet_collisions = np.sum([self.vehicles[k].num_of_collisions for k in range(self.number_of_vehicles)])
		# Compute the redundancy mask #
		self.redundancy_mask = np.sum([veh.detection_mask for veh in self.vehicles], axis=0)
		# Update the collective mask #
		self.collective_mask = self.redundancy_mask.astype(bool)
		# Update the historic visited mask #
		self.historic_visited_mask = np.logical_or(self.historic_visited_mask, self.collective_mask)
		# Update the isolation mask (for networked agents) #
		self.update_isolated_mask()

		return collision_array

	def update_isolated_mask(self):
		""" Compute the mask of isolated vehicles. Only for restricted fleets. """

		# Get the distance matrix #
		distance = self.get_distance_matrix()
		# Delete the diagonal (self-distance, always 0) #
		self.distance_between_agents = distance[~np.eye(distance.shape[0], dtype=bool)].reshape(distance.shape[0], -1)
		# True if all agents are further from the danger distance
		danger_of_isolation_mask = self.distance_between_agents > self.optimal_connection_distance
		self.danger_of_isolation = np.asarray([self.majority(value) for value in danger_of_isolation_mask])
		# True if all agents are further from the max connection distance
		isolation_mask = self.distance_between_agents > self.max_connection_distance
		self.isolated_mask = np.asarray([self.majority(value) for value in isolation_mask])
		self.number_of_disconnections += np.sum(self.isolated_mask)

	def measure(self, gt_field):

		"""
        Take a measurement in the given N positions
        :param gt_field:
        :return: An numpy array with dims (N,2)
        """
		positions = np.array([self.vehicles[k].position for k in range(self.number_of_vehicles)])

		values = []
		for pos in positions:
			values.append([gt_field[int(pos[0]), int(pos[1])]])

		if self.measured_locations is None:
			self.measured_locations = positions
			self.measured_values = values
		else:
			self.measured_locations = np.vstack((self.measured_locations, positions))
			self.measured_values = np.vstack((self.measured_values, values))

		return self.measured_values, self.measured_locations

	def reset(self, initial_positions=None):
		""" Reset the fleet """

		if initial_positions is None:
			initial_positions = self.initial_positions

		for k in range(self.number_of_vehicles):
			self.vehicles[k].reset(initial_position=initial_positions[k])

		self.agent_positions = np.asarray([veh.position for veh in self.vehicles])
		self.measured_values = None
		self.measured_locations = None
		self.fleet_collisions = 0
		self.number_of_disconnections = 0

		# Get the redundancy mask #
		self.redundancy_mask = np.sum([veh.detection_mask for veh in self.vehicles], axis=0)
		# Get the collective detection mask #
		self.collective_mask = self.redundancy_mask.astype(bool)
		self.historic_visited_mask = self.redundancy_mask.astype(bool)

		self.update_isolated_mask()

	def get_distances(self):
		return [self.vehicles[k].distance for k in range(self.number_of_vehicles)]

	def check_collisions(self, test_actions):
		""" Array of bools (True if collision) """
		return [self.vehicles[k].check_action(test_actions[k]) for k in range(self.number_of_vehicles)]

	def move_fleet_to_positions(self, goal_list):
		""" Move the fleet to the given positions.
         All goal positions must ve valid. """

		goal_list = np.atleast_2d(goal_list)

		for k in range(self.number_of_vehicles):
			self.vehicles[k].move_to_position(goal_position=goal_list[k])

	def get_distance_matrix(self):
		return distance_matrix(self.agent_positions, self.agent_positions)


class MultiAgentPatrolling(gym.Env):

	def __init__(self, scenario_map,
	             distance_budget,
	             number_of_vehicles,
	             fleet_initial_positions=None,
	             seed=0,
	             detection_length=2,
	             movement_length=2,
	             max_collisions=5,
	             forget_factor=1.0,
	             networked_agents=False,
	             max_connection_distance=10,
	             optimal_connection_distance=5,
	             max_number_of_disconnections=10,
	             attrittion=0.0,
	             obstacles=False,
	             hard_penalization=False):

		""" The gym environment """

		# Load the scenario map
		self.scenario_map = scenario_map
		self.visitable_locations = np.vstack(np.where(self.scenario_map != 0)).T
		self.number_of_agents = number_of_vehicles

		# Initial positions
		if fleet_initial_positions is None:
			self.random_inititial_positions = True
			random_positions_indx = np.random.choice(np.arange(0, len(self.visitable_locations)), number_of_vehicles, replace=False)
			self.initial_positions = self.visitable_locations[random_positions_indx]
		else:
			self.random_inititial_positions = False
			self.initial_positions = fleet_initial_positions

		self.obstacles = obstacles
	
		# Number of pixels
		self.distance_budget = distance_budget
		self.max_number_of_movements = distance_budget // detection_length
		# Number of agents
		self.seed = seed
		# Detection radius
		self.detection_length = detection_length
		self.forget_factor = forget_factor
		self.attrition = attrittion
		# Fleet of N vehicles
		self.optimal_connection_distance = optimal_connection_distance
		self.max_connection_distance = max_connection_distance

		# Create the fleets 
		self.fleet = DiscreteFleet(number_of_vehicles=self.number_of_agents,
		                           n_actions=8,
		                           fleet_initial_positions=self.initial_positions,
		                           movement_length=movement_length,
		                           navigation_map=self.scenario_map,
		                           max_connection_distance=self.max_connection_distance,
		                           optimal_connection_distance=self.optimal_connection_distance)

		self.max_collisions = max_collisions

		self.gt = GroundTruth(self.scenario_map, max_number_of_peaks=4, is_bounded=True, seed=self.seed)

		""" Model attributes """
		self.actual_known_map = None
		self.accumulated_known_map = None
		self.idleness_matrix = None
		self.importance_matrix = None
		self.inside_obstacles_map = None
		self.state = None
		self.fig = None

		self.action_space = gym.spaces.Discrete(8)
		self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(5, *self.scenario_map.shape),
		                                        dtype=np.float32)
		self.individual_action_state = gym.spaces.Discrete(8)

		self.networked_agents = networked_agents
		self.hard_networked_penalization = hard_penalization
		self.number_of_disconnections = 0
		self.max_number_of_disconnections = max_number_of_disconnections

		self.reward_normalization_value = self.fleet.vehicles[0].detection_mask

	def reset(self):
		""" Reset the environment """

		# Reset the ground truth #
		self.gt.reset()
		self.importance_matrix = self.gt.read()

		# Get the N random initial positions #
		if self.random_inititial_positions:
			random_positions_indx = np.random.choice(np.arange(0, len(self.visitable_locations)), self.number_of_agents, replace=False)
			self.initial_positions = self.visitable_locations[random_positions_indx]

		# Reset the positions of the fleet #
		self.fleet.reset(initial_positions=self.initial_positions)

		# New idleness mask (1-> high idleness, 0-> just visited)
		self.idleness_matrix = 1 - np.copy(self.fleet.collective_mask)

		# Randomly generated obstacles #
		if self.obstacles:
			# Generate a inside obstacles map #
			self.inside_obstacles_map = np.zeros_like(self.scenario_map)
			obstacles_pos_indx = np.random.choice(np.arange(0, len(self.visitable_locations)), size=20, replace=False)
			self.inside_obstacles_map[self.visitable_locations[obstacles_pos_indx, 0], self.visitable_locations[obstacles_pos_indx, 1]] = 1.0

			# Update the obstacle map for every agent #
			for i in range(self.number_of_agents):
				self.fleet.vehicles[i].navigation_map = self.scenario_map - self.inside_obstacles_map

		# Update the state of the agents #
		self.update_state()

		return self.state

	def update_temporal_mask(self):

		self.idleness_matrix = self.idleness_matrix + 1.0 / (self.forget_factor * self.max_number_of_movements)
		self.idleness_matrix = self.idleness_matrix - self.fleet.collective_mask
		self.idleness_matrix = np.clip(self.idleness_matrix, 0, 1)

		return self.idleness_matrix

	def update_information_importance(self):
		""" Applied the attrition term """
		self.importance_matrix = np.clip(
			self.importance_matrix - self.attrition * self.gt.read() * self.fleet.collective_mask, 0, 1)

	def update_state(self):

		state = np.zeros((3 + self.number_of_agents, *self.scenario_map.shape))

		# State 0 -> Known boundaries
		if self.obstacles:
			state[0] = self.scenario_map - np.logical_and(self.inside_obstacles_map, self.fleet.historic_visited_mask)
		else:
			state[0] = self.scenario_map

		# State 1 -> Temporal mask
		state[1] = self.idleness_matrix
		# State 2 -> Known information
		state[2] = self.importance_matrix * self.fleet.historic_visited_mask

		# State 3 and so on
		for i in range(self.number_of_agents):
			state[3 + i,
			      self.fleet.vehicles[i].position[0].astype(int),
			      self.fleet.vehicles[i].position[1].astype(int)] = 1.0

		self.state = state

	def step(self, action):

		# Process action movement
		collision_mask = self.fleet.move(action)

		# Compute reward
		reward = self.reward_function(collision_mask, action)

		# Update idleness and attrition
		self.update_temporal_mask()
		self.update_information_importance()

		# Update state
		self.update_state()

		# Final condition #
		done = np.mean(self.fleet.get_distances()) > self.distance_budget or self.fleet.fleet_collisions > self.max_collisions

		if self.networked_agents:

			if self.fleet.number_of_disconnections > self.max_number_of_disconnections and self.hard_networked_penalization:
				done = True

		return self.state, reward, done, {}

	def render(self, mode='human'):

		import matplotlib.pyplot as plt

		if self.fig is None:
			self.fig, self.axs = plt.subplots(1, 5)

			self.im0 = self.axs[0].imshow(self.state[0], cmap='gray')
			self.im1 = self.axs[1].imshow(self.state[1],  cmap='jet_r')
			self.im2 = self.axs[2].imshow(self.state[2],  cmap='coolwarm')
			self.im3 = self.axs[3].imshow(self.state[3], cmap='gray')
			self.im4 = self.axs[4].imshow(
				np.clip(np.sum(self.state[4:self.number_of_agents + 4][:, :, :, np.newaxis], axis=0), 0, 1),
				cmap='gray')

		self.im0.set_data(self.state[0])
		self.im1.set_data(self.state[1])
		self.im2.set_data(self.state[2])
		self.im3.set_data(self.state[3])
		self.im4.set_data(np.clip(np.sum(self.state[4:self.number_of_agents + 4][:, :, :, np.newaxis], axis=0), 0, 1))

		self.fig.canvas.draw()
		self.fig.canvas.flush_events()

		plt.draw()

		plt.pause(0.01)

	def individual_agent_observation(self, state=None, agent_num=0):

		if state is None:
			state = self.state
		assert 0 <= agent_num <= self.number_of_agents - 1, "Not enough agents for this observation request. "

		index = [0, 1, 2, 3 + agent_num]

		common_states = state[index]

		other_agents_positions_state = np.sum(
			state[np.delete(np.arange(3, 3 + self.number_of_agents), agent_num), :, :], axis=0)

		return np.concatenate((common_states, other_agents_positions_state[np.newaxis]), axis=0)

	def reward_function(self, collision_mask, actions):
		""" Reward function:
            r(t) = Sum(I(m)*W(m)/Dr(m)) - Pc - Pn
        """

		rewards = np.array(
			[np.sum(self.importance_matrix[veh.detection_mask.astype(bool)] * self.idleness_matrix[
				veh.detection_mask.astype(bool)] / (1 * self.detection_length * self.fleet.redundancy_mask[
				veh.detection_mask.astype(bool)])) for veh in self.fleet.vehicles]
		)

		cost = np.array([1 if action % 2 == 0 else np.sqrt(2) for action in actions]).astype(int)
		rewards = rewards / cost

		rewards[collision_mask] = -2.0

		if self.networked_agents:
			# For those agents that are too separated from the others (in danger of disconnection)
			min_distances = np.min(self.fleet.distance_between_agents[self.fleet.danger_of_isolation],
			                       axis=1) - self.fleet.optimal_connection_distance
			# Apply a penalization from 0 to -1 depending on the exceeding distance from the optimal
			rewards[self.fleet.danger_of_isolation] -= np.clip(
				min_distances / (self.max_connection_distance - self.optimal_connection_distance), 0, 1)

			rewards[self.fleet.isolated_mask] = -1.0

		return rewards

	def get_action_mask(self, ind=0):
		""" Return an array of Bools (True means this action for the agent ind causes a collision) """

		assert 0 <= ind < self.number_of_agents, 'Not enough agents!'

		return np.array(list(map(self.fleet.vehicles[ind].check_action, np.arange(0, 8))))


if __name__ == '__main__':

	sc_map = np.genfromtxt('Environment/example_map.csv', delimiter=',')

	initial_positions = np.array([[30, 20], [32, 20], [34, 20], [30, 22]])

	env = MultiAgentPatrolling(scenario_map=sc_map,
	                           fleet_initial_positions=initial_positions,
	                           distance_budget=200,
	                           number_of_vehicles=4,
	                           seed=10,
	                           detection_length=1,
	                           movement_length=1,
	                           max_collisions=500,
	                           forget_factor=0.5,
	                           attrittion=0.1,
	                           networked_agents=False,
	                           obstacles=True)

	env.reset()

	done = False

	R = []

	while not done:
		s, r, done, _ = env.step([env.action_space.sample() for _ in range(4)])
		env.render()
		print(r)
		R.append(r)
		env.individual_agent_observation(agent_num=0)

	R = np.asarray(R)
	env.render()
	plt.show()
	plt.close()
	plt.plot(np.cumsum(R, axis=0))
	plt.show()
