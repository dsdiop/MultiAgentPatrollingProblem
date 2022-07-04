from Environment.PatrollingEnvironments import MultiAgentPatrolling
import numpy as np
from Evaluation.Utils.metrics_wrapper import MetricsDataCreator
import matplotlib.pyplot as plt

def action2vector(action):
	""" Given 8 actions, compute the 2D vector of movement starting clockwise from 0. """
	if action == 0:
		return np.array([1, 0])
	elif action == 1:
		return np.array([1, 1])
	elif action == 2:
		return np.array([0, 1])
	elif action == 3:
		return np.array([-1, 1])
	elif action == 4:
		return np.array([-1, 0])
	elif action == 5:
		return np.array([-1, -1])
	elif action == 6:
		return np.array([0, -1])
	elif action == 7:
		return np.array([1, -1])
	else:
		raise ValueError('Invalid action')

def centroid_position(agents_positions):
	""" Given a list of agent positions, compute the centroid position. """
	return np.mean(agents_positions, axis=0)


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



N = 4
sc_map = np.genfromtxt('../../Environment/example_map.csv', delimiter=',')
initial_positions = np.asarray([[24, 21], [28, 24], [27, 19], [24, 24]])

env = MultiAgentPatrolling(scenario_map=sc_map,
                           fleet_initial_positions=initial_positions,
                           distance_budget=200,
                           number_of_vehicles=N,
                           seed=0,
                           detection_length=2,
                           movement_length=1,
                           max_collisions=50000,
                           forget_factor=0.5,
                           attrittion=0.1,
                           networked_agents=True,
                           hard_penalization=False,
                           max_connection_distance=7,
                           optimal_connection_distance=3,
                           max_number_of_disconnections=10,
                           obstacles=False)

metrics = MetricsDataCreator(metrics_names=['Accumulated Reward', 'Disconnections'],
                             algorithm_name='Greedy-Idleness fleet',
                             experiment_name='ResultsGreedyIdleness',
                             directory='./')

paths = MetricsDataCreator(metrics_names=['vehicle', 'x', 'y'],
                           algorithm_name='Greedy-Idleness fleet',
                           experiment_name='ResultsGreedyIdleness',
                           directory='./')

for run in range(10):

	s = env.reset()
	done = False

	centroid = centroid_position([veh.position for veh in env.fleet.vehicles])
	action,best = compute_greedy_action(centroid, (1+s[2])*s[1], 7, navigation_map=sc_map)
	actions = [action for _ in range(N)]

	step = 0
	R = 0
	t = 0
	recalculate = False

	# Initial register #
	metrics.register_step(run_num=run, step=step, metrics=[R, env.fleet.number_of_disconnections])
	for veh_id, veh in enumerate(env.fleet.vehicles):
		paths.register_step(run_num=run, step=step, metrics=[veh_id, veh.position[0], veh.position[1]])

	while not done:

		step += 1

		t += 1

		s, r, done, info = env.step(actions)

		centroid = centroid_position([veh.position for veh in env.fleet.vehicles])

		if any(env.fleet.check_collisions(actions)):
			action, best = compute_greedy_action(centroid, (1+s[2])*s[1], 7, navigation_map=sc_map)
			actions = np.asarray([action for _ in env.fleet.vehicles])

			if any(env.fleet.check_collisions(actions)):
				valid = False
				while not valid:
					new_action = env.action_space.sample()
					actions = np.asarray([new_action for _ in range(N)])
					collision_mask = env.fleet.check_collisions(actions)
					valid = not any(env.fleet.check_collisions(actions))
				recalculate = True
			else:
				recalculate = False


		R = np.mean(r) + R

		# Register positions and metrics #
		metrics.register_step(run_num=run, step=step, metrics=[R, env.fleet.number_of_disconnections])
		for veh_id, veh in enumerate(env.fleet.vehicles):
			paths.register_step(run_num=run, step=step, metrics=[veh_id, veh.position[0], veh.position[1]])

		env.render()
		plt.plot(best[1], best[0], 'rx')

metrics.register_experiment()
paths.register_experiment()
