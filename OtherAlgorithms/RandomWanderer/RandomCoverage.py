from Environment.PatrollingEnvironments import MultiAgentPatrolling
import numpy as np
from Evaluation.Utils.metrics_wrapper import MetricsDataCreator

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
                           max_collisions=5,
                           forget_factor=0.5,
                           attrittion=0.1,
                           networked_agents=True,
                           hard_penalization=False,
                           max_connection_distance=7,
                           optimal_connection_distance=3,
                           max_number_of_disconnections=10,
                           obstacles=False)

metrics = MetricsDataCreator(metrics_names=['Accumulated Reward', 'Disconnections'],
                             algorithm_name='Random Fleet Wandering',
                             experiment_name='RandomResultsNetworked',
                             directory='./')

paths = MetricsDataCreator(metrics_names=['vehicle', 'x', 'y'],
                           algorithm_name='Random Fleet Wandering',
                           experiment_name='RandomResultsNetworked_paths',
                           directory='./')

for run in range(10):

	env.reset()
	done = False
	valid = False
	while not valid:
		new_action = env.action_space.sample()
		action = np.asarray([new_action for _ in range(N)])
		collision_mask = env.fleet.check_collisions(action)
		valid = not any(env.fleet.check_collisions(action))

	step = 0
	R = 0

	# Initial register #
	metrics.register_step(run_num=run, step=step, metrics=[R, env.fleet.number_of_disconnections])
	for veh_id, veh in enumerate(env.fleet.vehicles):
		paths.register_step(run_num=run, step=step, metrics=[veh_id, veh.position[0], veh.position[1]])

	while not done:

		step += 1

		if any(env.fleet.check_collisions(action)):

			valid = False
			while not valid:
				new_action = env.action_space.sample()
				action = np.asarray([new_action for _ in range(N)])
				collision_mask = env.fleet.check_collisions(action)
				valid = not any(env.fleet.check_collisions(action))

		_, r, done, info = env.step(action)

		R = np.mean(r) + R

		# Register positions and metrics #
		metrics.register_step(run_num=run, step=step, metrics=[R, env.fleet.number_of_disconnections])
		for veh_id, veh in enumerate(env.fleet.vehicles):
			paths.register_step(run_num=run, step=step, metrics=[veh_id, veh.position[0], veh.position[1]])

		# env.render()

metrics.register_experiment()
paths.register_experiment()
