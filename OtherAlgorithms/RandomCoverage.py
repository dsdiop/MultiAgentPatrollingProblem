from Environment.PatrollingEnvironments import MultiAgentPatrolling
import numpy as np
import matplotlib.pyplot as plt

N = 4
sc_map = np.genfromtxt('../Environment/example_map.csv', delimiter=',')
initial_positions = np.asarray([[24,21],[28,24],[27,19],[24,24]])

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
                           networked_agents=False,
                           hard_penalization=False,
                           max_connection_distance=7,
                           optimal_connection_distance=3,
                           max_number_of_disconnections=10,
                           obstacles=False)

for t in range(10):

	env.reset()
	done = False

	action = np.asarray([env.action_space.sample() for _ in range(N)])
	while any(env.fleet.check_collisions(action)):
		action = np.asarray([env.action_space.sample() for _ in range(N)])

	R = []
	tt = 0
	R = 0
	while not done:

		tt += 1

		if any(env.fleet.check_collisions(action)):

			valid = False
			while not valid:
				new_actions = np.asarray([env.action_space.sample() for _ in range(N)])
				collision_mask = env.fleet.check_collisions(action)
				action[collision_mask] = new_actions[collision_mask]
				valid = not any(env.fleet.check_collisions(action))

		_, r, done, info = env.step(action)



		env.render()
		R += r
		print(env.fleet.get_distances())


# plt.show()
# plt.close()
# plt.plot(np.cumsum(R, axis=0))
# plt.show()
# plt.close()
