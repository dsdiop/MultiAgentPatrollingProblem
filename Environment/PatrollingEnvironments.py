import gym
import numpy as np
import matplotlib.pyplot as plt
from Environment.groundtruthgenerator import GroundTruth


class MultiagentDiscrete(gym.Space):
    r"""A multiagent discrete space in :math:`\{ 0, 1, \\dots, n-1 \}`.

    Example::

        >>> Discrete(2)

    """
    def __init__(self, n_actions, n_agents):
        assert n_actions >= 0
        self.n = n_actions
        self.n_agents = n_agents
        super(MultiagentDiscrete, self).__init__((), np.int64)

    def sample(self):

        return self.np_random.randint(self.n, size=(self.n_agents,))

    def __repr__(self):
        return "Multiagent Discrete(%d,%d)" % self.n, self.n_agents

    def __eq__(self, other):
        return isinstance(other, MultiagentDiscrete) and self.n_agents == other.n_agents


class DiscreteVehicle:

    def __init__(self, initial_position, n_actions, movement_length, navigation_map, detection_length):

        self.initial_position = initial_position
        self.position = np.copy(initial_position)
        self.waypoints = np.expand_dims(np.copy(initial_position), 0)
        self.trajectory = np.copy(self.waypoints)
        self.detection_length = detection_length
        self.navigation_map = navigation_map
        self.detection_map = self.compute_detection_map()

        self.distance = 0.0
        self.num_of_collisions = 0
        self.action_space = gym.spaces.Discrete(n_actions)
        self.angle_set = np.linspace(0, 2 * np.pi, n_actions, endpoint=False)
        self.movement_length = movement_length


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

        self.detection_map = self.compute_detection_map()

        return collide

    def check_collision(self, next_position):

        if self.navigation_map[int(next_position[0]), int(next_position[1])] == 0:
            return True  # There is a collision

        return False

    def update_trajectory(self):

        p1 = self.waypoints[-2]
        p2 = self.waypoints[-1]

        mini_traj = self.compute_trajectory_between_points(p1, p2)

        self.trajectory = np.vstack((self.trajectory, mini_traj))

    def compute_detection_map(self):

        known_mask = np.zeros_like(self.navigation_map)

        px, py = self.position

        # State - coverage area #
        x = np.arange(0, self.navigation_map.shape[0])
        y = np.arange(0, self.navigation_map.shape[1])

        # Compute the circular mask (area) of the state 3 #
        mask = (x[np.newaxis, :] - px) ** 2 + (y[:, np.newaxis] - py) ** 2 <= self.detection_length ** 2

        known_mask[mask.T] = 1.0

        return known_mask

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

        self.detection_map = self.compute_detection_map()

    def check_action(self, action):
        """ Return True if the action leads to a collision """

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
                                         navigation_map=navigation_map,
                                         detection_length=movement_length) for k in range(self.number_of_vehicles)]

        # Get the sum (logical or) of the detection maps #
        self.fleet_detection_map = self.vehicles[0].detection_map
        for n in range(1, self.number_of_vehicles):
            self.fleet_detection_map = np.logical_or(self.fleet_detection_map,
                                                     self.vehicles[n].detection_map)

        self.fleet_historic_visited_map = np.copy(self.fleet_detection_map)

        self.measured_values = None
        self.measured_locations = None
        self.fleet_collisions = 0

    def move(self, fleet_actions):

        collision_array = [self.vehicles[k].move(fleet_actions[k]) for k in range(self.number_of_vehicles)]

        self.fleet_collisions = np.sum([self.vehicles[k].num_of_collisions for k in range(self.number_of_vehicles)])

        # Get the sum (logical or) of the detection maps #
        self.fleet_detection_map = self.vehicles[0].detection_map
        for n in range(1, self.number_of_vehicles):
            self.fleet_detection_map = np.logical_or(self.fleet_detection_map,
                                                     self.vehicles[n].detection_map)

        self.fleet_historic_visited_map = np.logical_or(self.fleet_historic_visited_map, self.fleet_detection_map)

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

    def reset(self, initial_positions=None):

        if initial_positions is None:
            initial_positions = self.initial_positions

        for k in range(self.number_of_vehicles):
            self.vehicles[k].reset(initial_position=initial_positions[k])

        self.measured_values = None
        self.measured_locations = None
        self.fleet_collisions = 0

        # Get the sum (logical or) of the detection maps #
        self.fleet_detection_map = self.vehicles[0].detection_map
        for n in range(1, self.number_of_vehicles):
            self.fleet_detection_map = np.logical_or(self.fleet_detection_map,
                                                     self.vehicles[n].detection_map)

        self.fleet_historic_visited_map = np.copy(self.fleet_detection_map)

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


class MultiAgentPatrolling(gym.Env):

    def __init__(self, scenario_map, distance_budget,
                 number_of_vehicles, initial_positions=None, seed=0, detection_length=2, max_collisions = 5, forget_factor = 1):

        # Scenario map
        self.scenario_map = scenario_map
        self.visitable_locations = np.vstack(np.where(self.scenario_map != 0)).T

        # Initial positions
        if initial_positions is None:
            self.random_inititial_positions = True
            random_positions_indx = np.random.choice(np.arange(0,len(self.visitable_locations)), number_of_vehicles, replace = False)
            self.initial_positions = self.visitable_locations[random_positions_indx]
        else:
            self.random_inititial_positions = False
            self.initial_positions = initial_positions

        # Number of pixels
        self.distance_budget = distance_budget
        self.max_number_of_movements = distance_budget // detection_length
        # Number of agents
        self.num_agents = number_of_vehicles
        self.seed = seed
        # Detection radius
        self.detection_length = detection_length
        self.forget_factor = forget_factor
        # Fleet of N vehicles
        self.fleet = DiscreteFleet(number_of_vehicles=self.num_agents,
                                   n_actions=8,
                                   initial_positions=self.initial_positions,
                                   movement_length=self.detection_length, navigation_map=self.scenario_map)

        self.max_collisions = max_collisions * self.num_agents

        self.gt = GroundTruth(1 - self.scenario_map, 1, max_number_of_peaks=4, is_bounded=True, seed=self.seed)

        """ Model attributes """
        self.actual_known_map = None
        self.accumulated_known_map = None
        self.temporal_map = None
        self.inside_obstacles_map = None
        self.state = None
        self.fig = None

        self.action_space = MultiagentDiscrete(8, self.num_agents)
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(5, *self.scenario_map.shape), dtype=np.float32)
        self.individual_action_state = gym.spaces.Discrete(8)

    def reset(self):
        """ Reset the environment """

        # Reset the ground truth #
        self.gt.reset()
        # Reset the fleet
        
        if self.random_inititial_positions:
            random_positions_indx = np.random.choice(np.arange(0, len(self.visitable_locations)), self.num_agents, replace=False)
            self.initial_positions = self.visitable_locations[random_positions_indx]
            
        self.fleet.reset(initial_positions=self.initial_positions)

        # New temporal mask
        self.temporal_map = np.copy(self.fleet.fleet_detection_map)

        # New obstacles #
        self.inside_obstacles_map = np.zeros_like(self.scenario_map)
        obstacles_pos_indx = np.random.choice(np.arange(0, len(self.visitable_locations)), size=20, replace=False)
        self.inside_obstacles_map[
            self.visitable_locations[obstacles_pos_indx, 0], self.visitable_locations[obstacles_pos_indx, 1]] = 1.0
        # Update obstacles #
        for i in range(self.num_agents):
            self.fleet.vehicles[i].navigation_map = self.scenario_map - self.inside_obstacles_map

        self.update_state()

    def update_temporal_mask(self):

        self.temporal_map = self.temporal_map - 1.0 / (self.forget_factor * self.max_number_of_movements)
        self.temporal_map += self.fleet.fleet_detection_map

        self.temporal_map = np.clip(self.temporal_map, 0, 1)

        return self.temporal_map

    def update_state(self):

        state = np.zeros((3 + self.num_agents, *self.scenario_map.shape))

        # State 0 -> Known boundaries
        state[0] = self.scenario_map - np.logical_and(self.inside_obstacles_map, self.fleet.fleet_historic_visited_map)
        # State 1 -> Temporal mask
        state[1] = self.temporal_map
        # State 2 -> Known information
        state[2] = self.gt.read() * self.fleet.fleet_historic_visited_map

        # State 3 and so on
        for i in range(self.num_agents):
            state[3 + i, self.fleet.vehicles[i].position[0].astype(int), self.fleet.vehicles[i].position[1].astype(int)] = 1.0

        self.state = state

    def step(self, action):

        # Process action movement
        collision_mask = self.fleet.move(action)

        # Compute reward
        reward, individual_rewards = self.reward_function(collision_mask)

        # Update known map
        self.update_temporal_mask()

        # Update state
        self.update_state()

        # Final condition #
        done = np.mean(self.fleet.get_distances()) > self.distance_budget or self.fleet.fleet_collisions > self.max_collisions

        info = {'individual_rewards': individual_rewards}

        return self.state, reward, done, info

    def render(self):

        import matplotlib.pyplot as plt

        if self.fig is None:

            self.fig, self.axs = plt.subplots(1, 5)

            self.im0 = self.axs[0].imshow(self.state[0], cmap='gray')
            self.im1 = self.axs[1].imshow(self.state[1], interpolation = 'bicubic', cmap='jet')
            self.im2 = self.axs[2].imshow(self.state[2], interpolation = 'bicubic', cmap='coolwarm')
            self.im3 = self.axs[3].imshow(self.state[3], cmap='gray')
            self.im4 = self.axs[4].imshow(np.clip(np.sum(self.state[4:self.num_agents + 4][:, :, :, np.newaxis], axis=0), 0, 1), cmap='gray')


        self.im0.set_data(self.state[0])
        self.im1.set_data(self.state[1])
        self.im2.set_data(self.state[2])
        self.im3.set_data(self.state[3])
        self.im4.set_data(np.clip(np.sum(self.state[4:self.num_agents + 4][:, :, :, np.newaxis], axis=0), 0, 1))

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        plt.pause(0.1)

    def individual_agent_observation(self, state = None, agent_num=0):

        if state is None:
            state = self.state
        assert 0 <= agent_num <= self.num_agents - 1, "Not enough agents for this observation request. "

        index = [0, 1, 2, 3 + agent_num]

        common_states = state[index]

        other_agents_positions_state = np.sum(state[np.delete(np.arange(3, 3 + self.num_agents), agent_num), :, :], axis=0)

        return np.concatenate((common_states, other_agents_positions_state[np.newaxis]), axis = 0)

    def reward_function(self, collision_mask):


        individual_rewards = np.array([np.sum(veh.detection_map * (1-self.temporal_map))/(np.pi*self.detection_length**2) for veh in self.fleet.vehicles])

        collective_reward = np.mean(individual_rewards)

        total_rewards = individual_rewards + collective_reward

        total_rewards[collision_mask] = -1.0


        return total_rewards, individual_rewards

    def get_action_mask(self, ind = 0):
        """ Return an array of Bools (True means this action for the agent ind causes a collision) """

        assert 0 <= ind < self.num_agents, 'Not enough agents!'

        return np.array(list(map(self.fleet.vehicles[ind].check_action, np.arange(0, 8))))

if __name__ == '__main__':

    sc_map = np.genfromtxt('example_map.csv', delimiter=',')

    initial_positions = np.array([[30, 20], [40, 20], [20, 20], [10, 20]])

    env = MultiAgentPatrolling(scenario_map=sc_map, initial_positions=initial_positions, distance_budget=200,
                               number_of_vehicles=4, seed=0, detection_length=2, max_collisions=10000, forget_factor = 0.5)

    env.reset()

    done = False

    R = []

    while not done:

        s,r,done,_ = env.step(env.action_space.sample())
        env.render()
        print(r)
        R.append(r)
        env.individual_agent_observation(agent_num=0)

    R = np.asarray(R)
    env.render()
    plt.show()
    plt.close()
    plt.plot(np.cumsum(R,axis=0))
    plt.show()
