from Environment.PatrollingEnvironments import MultiAgentPatrolling
import numpy as np
from Evaluation.Utils.metrics_wrapper import MetricsDataCreator
import matplotlib.pyplot as plt


LEFT = 1
DOWN = 2
RIGHT = 3
TOP = 4

GOING_DOWN = 0
GOING_UP = 1


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
                             algorithm_name='Lawnmower',
                             experiment_name='ResultsLM',
                             directory='./')

paths = MetricsDataCreator(metrics_names=['vehicle', 'x', 'y'],
                           algorithm_name='Lawnmower',
                           experiment_name='ResultsLM_paths',
                           directory='./')

def centroid_position(agents_positions):
	""" Given a list of agent positions, compute the centroid position. """
	return np.mean(agents_positions, axis=0)

for run in range(10):

	s = env.reset()
	done = False

	v_direction = GOING_UP
	h_direction = LEFT
	action = 2
	troubled_path_finded = False

	step = 0
	R = 0

	# Initial register #
	metrics.register_step(run_num=run, step=step, metrics=[R, env.fleet.number_of_disconnections])
	for veh_id, veh in enumerate(env.fleet.vehicles):
		paths.register_step(run_num=run, step=step, metrics=[veh_id, veh.position[0], veh.position[1]])

	while not done:

		step += 1


		actions = [action for _ in range(N)]
		s, r, done, info = env.step(actions)

		positions = np.asarray([veh.position for veh in env.fleet.vehicles])
		print(np.max(positions[:,0]))

		if any(env.fleet.check_collisions(actions)):


			positions = np.asarray([veh.position for veh in env.fleet.vehicles])
			print(np.min(positions[:,0]))
			if np.max(positions[:,0]) > 0.85*sc_map.shape[0] and v_direction == GOING_DOWN:
				v_direction = GOING_UP
				h_direction = LEFT
			elif np.min(positions[:,0]) < 0.15*sc_map.shape[0] and v_direction == GOING_UP:
				print("HERE!")
				v_direction = GOING_DOWN
				h_direction = RIGHT


			# In case of collision, change the direction:
			if v_direction == GOING_DOWN:
				if h_direction == RIGHT:
					action = 7
					h_direction = DOWN
				elif h_direction == DOWN:
					action = 2
					h_direction = RIGHT
			else:
				if h_direction == LEFT:
					action = 6
					h_direction = TOP
				elif h_direction == TOP:
					action = 3
					h_direction = LEFT

		R = np.mean(r) + R

		# Register positions and metrics #
		metrics.register_step(run_num=run, step=step, metrics=[R, env.fleet.number_of_disconnections])
		for veh_id, veh in enumerate(env.fleet.vehicles):
			paths.register_step(run_num=run, step=step, metrics=[veh_id, veh.position[0], veh.position[1]])

		#env.render()

metrics.register_experiment()
paths.register_experiment()
