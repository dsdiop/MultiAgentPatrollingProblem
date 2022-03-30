from Environment.PatrollingEnvironments import MultiAgentPatrolling
from Algorithm.RainbowDQL.Agent.DuelingDQNAgent import MultiAgentDuelingDQNAgent
import numpy as np

N = 4
sc_map = np.genfromtxt('../Environment/example_map.csv', delimiter=',')
visitable_locations = np.vstack(np.where(sc_map != 0)).T
random_index = np.random.choice(np.arange(0, len(visitable_locations)), N, replace=False)
initial_positions = visitable_locations[random_index]

env = MultiAgentPatrolling(scenario_map=sc_map, initial_positions=initial_positions, distance_budget=200,
                           number_of_vehicles=N, seed=0, detection_length=2, max_collisions=5, forget_factor=0.5)

multiagent = MultiAgentDuelingDQNAgent(env=env,
                                       memory_size=int(1E5),
                                       batch_size=64,
                                       target_update=1,
                                       soft_update=True,
                                       tau=0.0001,
                                       epsilon_values=[1.0, 0.05],
                                       epsilon_interval=[0.0, 0.33],
                                       learning_starts=10,
                                       gamma=0.99,
                                       lr=1e-4,
                                       noisy=False,
                                       safe_actions=False)


multiagent.train(episodes=10000)
