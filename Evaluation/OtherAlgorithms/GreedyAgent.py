import numpy as np
import pandas as pd 
from tqdm import trange
import matplotlib.pyplot as plt

from Evaluation.Utils.metrics_wrapper import MetricsDataCreator
from Evaluation.Utils.anneal_nu import anneal_nu
import os

class GreedyAgent:

	def __init__(self, world: np.ndarray, movement_length: float, detection_length: float, number_of_actions: int, seed = 0):
		
		self.world = world
		self.move_length = movement_length
		self.detection_length = detection_length
		self.number_of_actions = number_of_actions
		self.seed = seed
		self.rng = np.random.default_rng(seed=self.seed)
	
	def move(self, actual_position, other_positions, interest_map):

		# Compute if there is an obstacle or reached the border #
		new_possible_positions = [actual_position + self.action_to_vector(i) for i in range(self.number_of_actions)]
		OBS = self.check_possible_collisions(new_possible_positions, other_positions)
		if OBS.all():
			return 0, actual_position
		interest_recollected = [-np.inf if OBS[i] else self.interest_recollected(new_possible_positions[i], interest_map) for i in range(len(OBS))]
		action = self.rng.choice(np.where(np.array(interest_recollected) == np.max(interest_recollected))[0]) # If there are more than one action with the same interest, choose randomly
		return action , actual_position + self.action_to_vector(action)
	
	def interest_recollected(self, agent_position, interest_map):
		""" Given the agent position and the interest map, compute the interest recollected. """

		masked_interest_map = interest_map * self.compute_detection_mask(agent_position) * self.world
		interest_recollected = np.sum(masked_interest_map)
		return interest_recollected

	def compute_detection_mask(self, agent_position):
		""" Compute the circular mask """
  
		px, py = agent_position.astype(int)
  		# State - coverage area #
		x = np.arange(0, self.world.shape[0])
		y = np.arange(0, self.world.shape[1])

		# Compute the circular mask (area) #
		mask = (x[np.newaxis, :] - px) ** 2 + (y[:, np.newaxis] - py) ** 2 <= self.detection_length ** 2

		known_mask = np.zeros_like(self.world)
		known_mask[mask.T] = 1.0
		return known_mask

	def action_to_vector(self, action):
		""" Transform an action to a vector """
		angle_set = np.linspace(0, 2 * np.pi, self.number_of_actions, endpoint=False)
		angle = angle_set[action]
		movement = np.round(np.array([self.move_length * np.cos(angle), self.move_length * np.sin(angle)])).astype(int)
		return movement.astype(int)

	def check_possible_collisions(self, new_possible_positions, other_positions):
		""" Check if the agent collides with an obstacle """
		agent_collisions = [list(elemento) in other_positions for elemento in new_possible_positions]
		world_collisions = [(new_position[0] < 0) or (new_position[0] >= self.world.shape[0]) or (new_position[1] < 0) or (new_position[1] >= self.world.shape[1])
                      		for new_position in new_possible_positions]
		border_collisions = [self.world[new_position[0], new_position[1]] == 0 for new_position in new_possible_positions]
		OBS = np.logical_or(agent_collisions, np.logical_or(world_collisions,border_collisions))	
		return OBS


def run_evaluation(path: str, env, algorithm: str, runs: int, n_agents: int, ground_truth_type: str, render = False):

	metrics = MetricsDataCreator(metrics_names=['Policy Name',
											'Accumulated Reward Intensification',
											'Accumulated Reward Exploration',
											'Total Accumulated Reward',
											'Total Length',
											'Total Collisions',
											'Average global idleness Intensification',
											'Average global idleness Exploration',
											'Sum global idleness Intensification',
											'Percentage Visited Exploration',
											'Percentage Visited'],
							algorithm_name=algorithm,
							experiment_name=f'{algorithm}_Results',
							directory=path)
	if os.path.exists(path + algorithm + '_Results.csv'):
		metrics.load_df(path + algorithm + '_Results.csv')
        
	paths = MetricsDataCreator(metrics_names=['vehicle', 'x', 'y'],
                            algorithm_name=algorithm,
                            experiment_name=f'{algorithm}_paths',
                            directory=path)
    
	if os.path.exists(path + algorithm + '_paths.csv'):
		paths.load_df(path + algorithm + '_paths.csv')
  

	greedy_agents = [GreedyAgent(world = env.scenario_map, number_of_actions = 8, movement_length = env.movement_length, detection_length=env.detection_length, seed=0) for i in range(n_agents)]

	
	distance_budget = env.distance_budget

	for run in trange(runs):
		# Increment the step counter #
		step = 0
		
		# Reset the environment #
		s = env.reset()

		if render:
			env.render()

		# Reset dones #
		done = {agent_id: False for agent_id in range(env.number_of_agents)}
		#plt.savefig(f'{path}_{algorithm}_{run}.png')
		# Update the metrics #
		total_reward = 0
		total_reward_information = 0
		total_reward_exploration = 0
		total_length = 0
		total_collisions = 0
		percentage_visited = np.count_nonzero(env.fleet.historic_visited_mask) / np.count_nonzero(env.scenario_map)
		percentage_visited_exp = np.count_nonzero(env.fleet.historic_visited_mask) / np.count_nonzero(env.scenario_map)
		average_global_idleness_exp = env.average_global_idleness_exp
		sum_global_interest = env.sum_global_idleness
		sum_instantaneous_global_idleness = 0
		steps_int = 0
		average_global_idleness_int = sum_instantaneous_global_idleness
		metrics_list = [algorithm, total_reward_information,
						total_reward_exploration,
						total_reward, total_length,
						total_collisions,
						average_global_idleness_int,
						average_global_idleness_exp,
						sum_global_interest,
						percentage_visited_exp,
						percentage_visited]
		# Initial register #
		metrics.register_step(run_num=run, step=total_length, metrics=metrics_list)
		for veh_id, veh in enumerate(env.fleet.vehicles):
			paths.register_step(run_num=run, step=total_length, metrics=[veh_id, veh.position[0], veh.position[1]])

		while not all(done.values()):

			total_length += 1
			other_positions = []
			acts = []
			interest_map =  s[np.argmax(env.active_agents)][1]*np.clip(s[np.argmax(env.active_agents)][2],env.minimum_importance,1)
			# Compute the actions #
			for i in range(n_agents):
				action, new_position = greedy_agents[i].move(env.fleet.vehicles[i].position, other_positions, interest_map)
				acts.append(action)
				if list(new_position) in other_positions:
					OBS = False
				other_positions.append(list(new_position))
				interest_map =  np.clip(interest_map - greedy_agents[i].compute_detection_mask(new_position),env.minimum_importance,1)
			actions = {i: acts[i] for i in range(n_agents)}
			# Process the agent step #
			s, reward, done, _ = env.step(actions)

			if render:
				env.render()
			distance = np.min([np.max(env.fleet.get_distances()), distance_budget])
			nu = anneal_nu(p= distance / distance_budget)
			rewards = np.asarray(list(reward.values()))
			percentage_visited = np.count_nonzero(env.fleet.historic_visited_mask) / np.count_nonzero(env.scenario_map)
			if nu<0.5:
				steps_int += 1
				sum_instantaneous_global_idleness += env.instantaneous_global_idleness
				average_global_idleness_int = sum_instantaneous_global_idleness/steps_int
				total_reward_information += np.sum(rewards[:,0])
			else:
				average_global_idleness_exp = np.copy(env.average_global_idleness_exp)
				percentage_visited_exp = np.count_nonzero(env.fleet.historic_visited_mask) / np.count_nonzero(env.scenario_map)
				total_reward_exploration += np.sum(rewards[:,1])

			total_collisions += env.fleet.fleet_collisions    
			total_reward = total_reward_exploration + total_reward_information


			sum_global_interest = env.sum_global_idleness
			metrics_list = [algorithm, total_reward_information,
							total_reward_exploration,
							total_reward, total_length,
							total_collisions,
							average_global_idleness_int,
							average_global_idleness_exp,
							sum_global_interest,
							percentage_visited_exp,
							percentage_visited]
			metrics.register_step(run_num=run, step=total_length, metrics=metrics_list)
			for veh_id, veh in enumerate(env.fleet.vehicles):
				paths.register_step(run_num=run, step=total_length, metrics=[veh_id, veh.position[0], veh.position[1]])

		


	if not render:
		metrics.register_experiment()
		paths.register_experiment()
	else:
		plt.close()



def compute_greedy_action(agent_position, interest_map, max_distance, navigation_map):

	""" Given the agent position and the interest map, compute the greedy action. """

	px, py = agent_position.astype(int)

	# State - coverage area #
	x = np.arange(0, navigation_map.shape[0])
	y = np.arange(0, navigation_map.shape[1])

	# Compute the circular mask (area) #
	mask = (x[np.newaxis, :] - px) ** 2 + (y[:, np.newaxis] - py) ** 2 <= max_distance ** 2

	known_mask = np.zeros_like(navigation_map)
	known_mask[mask.T] = 1.0

	masked_interest_map = interest_map * known_mask * navigation_map

	# Compute the action that moves the agent in the direction of the maximum value of masked_interest_map #
	best_position_x, best_position_y = np.unravel_index(np.argmax(masked_interest_map), masked_interest_map.shape)

	direction = np.arctan2(best_position_y - py, best_position_x - px)

	direction = direction + 2*np.pi if direction < 0 else direction

	greedy_action = np.argmin(np.abs(direction - np.linspace(0, 2*np.pi, 8)))

	return greedy_action, np.asarray([best_position_y, best_position_x])


