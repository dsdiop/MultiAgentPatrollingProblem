import gym
import numpy as np
import matplotlib.pyplot as plt
from groundtruthgenerator import GroundTruth


class DiscreteVehicle:

	def __init__(self, initial_position, n_actions, movement_length, navigation_map):

		self.initial_position = initial_position
		self.position = np.copy(initial_position)
		self.waypoints = np.expand_dims(np.copy(initial_position), 0)
		self.trajectory = np.copy(self.waypoints)

		self.distance = 0.0
		self.num_of_collisions = 0
		self.action_space = gym.spaces.Discrete(n_actions)
		self.angle_set = np.linspace(0, 2 * np.pi, n_actions, endpoint=False)
		self.movement_length = movement_length
		self.navigation_map = navigation_map

	def move(self, action):

		self.distance += self.movement_length
		angle = self.angle_set[action]
		movement = np.array([self.movement_length * np.cos(angle), self.movement_length * np.sin(angle)])
		next_position = self.position + movement

		if self.check_collision(next_position):
			collide = True
			self.num_of_collisions += 1
		else:
			collide = False
			self.position = next_position
			self.waypoints = np.vstack((self.waypoints, [self.position]))
			self.update_trajectory()

		return collide

	def check_collision(self, next_position):

		if self.navigation_map[int(next_position[0]), int(next_position[1])] == 0:
			return True
		return False

	def update_trajectory(self):

		p1 = self.waypoints[-2]
		p2 = self.waypoints[-1]

		mini_traj = self.compute_trajectory_between_points(p1, p2)

		self.trajectory = np.vstack((self.trajectory, mini_traj))

	@staticmethod
	def compute_trajectory_between_points(p1, p2):
		trajectory = None

		p = p1.astype(int)
		d = p2.astype(int) - p1.astype(int)
		N = np.max(np.abs(d))
		s = d / N

		for ii in range(0, N):
			p = p + s
			if trajectory is None:
				trajectory = np.array([np.rint(p)])
			else:
				trajectory = np.vstack((trajectory, [np.rint(p)]))

		return trajectory.astype(int)

	def reset(self, initial_position):

		self.initial_position = initial_position
		self.position = np.copy(initial_position)
		self.waypoints = np.expand_dims(np.copy(initial_position), 0)
		self.trajectory = np.copy(self.waypoints)
		self.distance = 0.0
		self.num_of_collisions = 0

	def check_action(self, action):

		angle = self.angle_set[action]
		movement = np.array([self.movement_length * np.cos(angle), self.movement_length * np.sin(angle)])
		next_position = self.position + movement

		return self.check_collision(next_position)

	def move_to_position(self, goal_position):

		""" Add the distance """
		assert self.navigation_map[goal_position[0], goal_position[1]] == 1, "Invalid position to move"
		self.distance += np.linalg.norm(goal_position - self.position)
		""" Update the position """
		self.position = goal_position


class DiscreteFleet:

	def __init__(self, number_of_vehicles, n_actions, initial_positions, movement_length, navigation_map):

		self.number_of_vehicles = number_of_vehicles
		self.initial_positions = initial_positions
		self.n_actions = n_actions
		self.movement_length = movement_length
		self.vehicles = [DiscreteVehicle(initial_position=initial_positions[k],
		                                 n_actions=n_actions,
		                                 movement_length=movement_length,
		                                 navigation_map=navigation_map) for k in range(self.number_of_vehicles)]

		self.measured_values = None
		self.measured_locations = None
		self.fleet_collisions = 0

	def move(self, fleet_actions):

		collision_array = [self.vehicles[k].move(fleet_actions[k]) for k in range(self.number_of_vehicles)]

		self.fleet_collisions = np.sum([self.vehicles[k].num_of_collisions for k in range(self.number_of_vehicles)])

		return collision_array

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

	def reset(self, initial_positions = None):

		if initial_positions is None:
			initial_positions = self.initial_positions

		for k in range(self.number_of_vehicles):
			self.vehicles[k].reset(initial_position=initial_positions[k])

		self.measured_values = None
		self.measured_locations = None
		self.fleet_collisions = 0

	def get_distances(self):

		return [self.vehicles[k].distance for k in range(self.number_of_vehicles)]

	def check_collisions(self, test_actions):

		return [self.vehicles[k].check_action(test_actions[k]) for k in range(self.number_of_vehicles)]

	def move_fleet_to_positions(self, goal_list):
		""" Move the fleet to the given positions.
         All goal positions must ve valid. """

		goal_list = np.atleast_2d(goal_list)

		for k in range(self.number_of_vehicles):
			self.vehicles[k].move_to_position(goal_position=goal_list[k])


class MultiAgentPatrolling(gym.Env):

	def __init__(self, scenario_map, initial_positions, distance_budget,
	             number_of_vehicles, seed=0, detection_length=2):

		# Scenario map
		self.scenario_map = scenario_map
		self.visitable_locations = np.vstack(np.where(self.scenario_map != 0)).T
		# Initial positions
		self.initial_positions = initial_positions
		# Number of pixels
		self.distance_budget = distance_budget
		self.max_number_of_movements = distance_budget // detection_length
		# Number of agents
		self.num_vehicles = number_of_vehicles
		self.seed = seed
		# Detection radius
		self.detection_length = detection_length
		# Fleet of N vehicles
		self.fleet = DiscreteFleet(number_of_vehicles=self.num_vehicles,
		                           n_actions=8,
		                           initial_positions=self.initial_positions,
		                           movement_length=self.detection_length, navigation_map=self.scenario_map)

		self.gt = GroundTruth(1 - self.scenario_map, 1, max_number_of_peaks=4, is_bounded=True, seed=self.seed)


		""" Model attributes """
		self.actual_known_map = None
		self.accumulated_known_map = None
		self.temporal_map = None
		self.inside_obstacles_map = None
		self.state = None
		self.fig = None


	def reset(self):
		""" Reset the environment """

		# Reset the ground truth #
		self.gt.reset()
		# Reset the fleet
		self.fleet.reset()

		# Get the known mask #
		self.accumulated_known_map = np.zeros_like(self.scenario_map)
		self.actual_known_map = np.zeros_like(self.scenario_map)
		self.actual_known_map = self.known_mask()
		self.accumulated_known_map = np.logical_or(self.accumulated_known_map, self.actual_known_map)

		# New temporal mask
		self.temporal_map = np.copy(self.actual_known_map)
		# New obstacles #
		self.inside_obstacles_map = np.zeros_like(self.scenario_map)
		obstacles_pos_indx = np.random.choice(np.arange(0,len(self.visitable_locations)), size = 20, replace=False)
		self.inside_obstacles_map[self.visitable_locations[obstacles_pos_indx, 0], self.visitable_locations[obstacles_pos_indx, 1]] = 1.0
		# Update obstacles #
		for i in range(self.num_vehicles):
			self.fleet.vehicles[i].navigation_map = self.scenario_map - self.inside_obstacles_map

		self.update_state()

	def known_mask(self):
		""" Process the known mask """

		known_map = np.zeros_like(self.scenario_map)

		for vehicle in self.fleet.vehicles:

			px, py = vehicle.position

			# State - coverage area #
			x = np.arange(0, self.scenario_map.shape[0])
			y = np.arange(0, self.scenario_map.shape[1])

			# Compute the circular mask (area) of the state 3 #
			mask = (x[np.newaxis, :] - px) ** 2 + (y[:, np.newaxis] - py) ** 2 <= self.detection_length ** 2

			known_map[mask.T] = 1.0

		return known_map

	def update_temporal_mask(self):

		self.temporal_map = self.temporal_map - 1/self.max_number_of_movements
		self.temporal_map += self.actual_known_map

		self.temporal_map = np.clip(self.temporal_map, 0, 1)

		return self.temporal_map

	def update_masks(self):

		self.actual_known_map = self.known_mask()
		self.accumulated_known_map = np.logical_or(self.accumulated_known_map, self.actual_known_map)
		self.update_temporal_mask()

	def update_state(self):

		state = np.zeros((3 + self.num_vehicles, *self.scenario_map.shape))

		# State 0 -> Known boundaries
		state[0] = self.scenario_map - np.logical_and(self.inside_obstacles_map ,self.accumulated_known_map)
		# State 1 -> Temporal mask
		state[1] = self.temporal_map
		# State 2 -> Known information
		state[2] = self.gt.read() * self.accumulated_known_map

		# State 3 and so on
		for i in range(self.num_vehicles):
			state[3+i, self.fleet.vehicles[i].position[0].astype(int), self.fleet.vehicles[i].position[1].astype(int)] = 1.0

		self.state = state

	def step(self, action):

		# Process action movement
		self.fleet.move(action)

		# Update known map
		self.update_masks()

		# Update state
		self.update_state()

	def render(self):

		import matplotlib.pyplot as plt

		cmaps = ['gray', 'jet', 'coolwarm']
		[cmaps.append('gray') for i in range(self.num_vehicles)]

		if self.fig is None:
			self.fig, self.axs = plt.subplots(1,len(self.state))
			self.im = []

			for i in range(len(self.state)):
				self.im.append(self.axs[i].imshow(self.state[i], cmap = cmaps[i]))

		for i in range(len(self.state)):
			self.im[i].set_data(self.state[i])

		self.fig.canvas.draw()
		self.fig.canvas.flush_events()

		plt.pause(0.1)

	def individual_agent_observation(self, agent_num = 0):

		assert 0 <= agent_num <= self.num_vehicles - 1, "Not enough agents for this observation request. "
		index = [0,1,2,3 + agent_num]
		return self.state[index]


if __name__ == '__main__':

	sc_map = np.genfromtxt('example_map.csv', delimiter=',')

	initial_positions = np.array([[30,20], [40,20], [20,20]])

	env = MultiAgentPatrolling(scenario_map=sc_map, initial_positions = initial_positions, distance_budget = 100,
	             number_of_vehicles = 3, seed=0, detection_length=2)

	env.reset()

	for i in range(100):

		env.step(np.random.randint(0,8, size=(env.num_vehicles,)))
		env.render()

