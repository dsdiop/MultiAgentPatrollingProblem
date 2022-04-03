from Environment.PatrollingEnvironments import MultiAgentPatrolling
import numpy as np
import matplotlib.pyplot as plt

N = 5
sc_map = np.genfromtxt('../Environment/ypacarai_map.csv')
visitable_locations = np.vstack(np.where(sc_map != 0)).T
random_index = np.random.choice(np.arange(0, len(visitable_locations)), N, replace=False)
# initial_positions = visitable_locations[random_index]

env = MultiAgentPatrolling(scenario_map=sc_map, initial_positions=None, distance_budget=1000,
                           number_of_vehicles=N, seed=0, detection_length=4, max_collisions=500, forget_factor=0.5)

for t in range(1):

	env.reset()
	done = False

	action = env.action_space.sample()
	while any(env.fleet.check_collisions(action)):
		action = env.action_space.sample()

	R = []

	while not done:

		_, r, done, info = env.step(action)

		if any(env.fleet.check_collisions(action)):

			valid = False

			while not valid:
				new_actions = env.action_space.sample()
				invalid_mask = env.fleet.check_collisions(action)
				action[invalid_mask] = new_actions[invalid_mask]
				valid = not any(env.fleet.check_collisions(action))

		print("Reward")
		print(r)
		R.append(info['individual_rewards'])
		#env.render()

	env.render()
	plt.show()
	plt.close()
	plt.plot(np.cumsum(R, axis=0))
	plt.show()
	plt.close()
