from Environment.PatrollingEnvironments import MultiAgentPatrolling
from Algorithm.RainbowDQL.Agent.DuelingDQNAgent import MultiAgentDuelingDQNAgent
import numpy as np

N = 4
sc_map = np.genfromtxt('../Environment/Maps/example_map.csv', delimiter=',')
visitable_locations = np.vstack(np.where(sc_map != 0)).T
random_index = np.random.choice(np.arange(0,len(visitable_locations)), N, replace=False)
initial_positions = np.asarray([[24, 21],[28,24],[27,19],[24,24]])

#initial_positions = np.array([[30, 20], [32, 20], [34, 20], [30, 22]])[:N, :]
#visitable = np.column_stack(np.where(sc_map == 1))
# initial_positions = visitable[np.random.randint(0,len(visitable), size=N), :]


env = MultiAgentPatrolling(scenario_map=sc_map,
                           fleet_initial_positions=initial_positions,
                           distance_budget=250,
                           number_of_vehicles=N,
                           seed=0,
                           miopic=True,
                           detection_length=1,
                           movement_length=1,
                           max_collisions=20,
                           forget_factor=0.5,
                           attrittion=0.1,
                           networked_agents=False,
                           ground_truth_type='algae_bloom',
                           obstacles=True,
                           frame_stacking=1,
                           state_index_stacking=(2, 3, 4),
                           reward_weights=(1.0, 0.1)
                           )

multiagent = MultiAgentDuelingDQNAgent(env=env,
                                       memory_size=int(1E6),
                                       batch_size=64,
                                       target_update=1000,
                                       soft_update=False,
                                       tau=0.0001,
                                       epsilon_values=[1.0, 0.1],
                                       epsilon_interval=[0.0, 0.33],
                                       learning_starts=0,
                                       gamma=0.99,
                                       lr=1e-4,
                                       noisy=False,
                                       train_every=10,
                                       save_every=5000,
                                       distributional=False,
                                       use_nu=True,
                                       nu_intervals=[[0., 1], [0.25, 0.75], [0.75, 0.25], [1., 0.]])

multiagent.train(episodes=100000)
