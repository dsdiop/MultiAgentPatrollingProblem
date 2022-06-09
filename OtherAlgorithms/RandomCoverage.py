from Environment.PatrollingEnvironments import MultiAgentPatrolling
import numpy as np
import matplotlib.pyplot as plt


N = 4
sc_map = np.genfromtxt('../Environment/example_map.csv', delimiter=',')
visitable_locations = np.vstack(np.where(sc_map != 0)).T
random_index = np.random.choice(np.arange(0,len(visitable_locations)), N, replace=False)
initial_positions = visitable_locations[random_index]

env = MultiAgentPatrolling(scenario_map=sc_map, fleet_initial_positions=initial_positions, distance_budget=200,
                           number_of_vehicles=N, seed=0, detection_length=2, max_collisions=5, forget_factor=0.5,
                           attrittion=0.1, networked_agents=False, max_connection_distance=20, min_isolation_distance=10,
                           max_number_of_disconnections=50)

for t in range(1):
	env.reset()
	done = False

	action = env.action_space.sample()
	while any(env.fleet.check_collisions(action)):
		action = env.action_space.sample()

	R = []

	while not done:

		print(env.fleet.danger_of_isolation)

		_, r, done, info = env.step(action)

		if any(env.fleet.check_collisions(action)):

			valid = False

			while not valid:
				new_actions = env.action_space.sample()
				invalid_mask = env.fleet.check_collisions(action)
				action[invalid_mask] = new_actions[invalid_mask]
				valid = not any(env.fleet.check_collisions(action))

		print("Reward")
		print(env.number_of_disconnections)

		env.render()

	#plt.show()
	#plt.close()
	#plt.plot(np.cumsum(R, axis=0))
	#plt.show()
	#plt.close()
