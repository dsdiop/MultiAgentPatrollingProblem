from Environment.PatrollingEnvironments import MultiAgentPatrolling
from Algorithm.RainbowDQL.Agent.DuelingDQNAgent import MultiAgentDuelingDQNAgent
import numpy as np
import matplotlib.pyplot as plt

N = 4
sc_map = np.genfromtxt('../Environment/example_map.csv', delimiter=',')

env = MultiAgentPatrolling(scenario_map=sc_map, initial_positions=None, distance_budget=200,
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
                                       noisy=True,
                                       safe_actions=True)


multiagent.load_model('/home/azken/Samuel/MultiAgentPatrollingProblem/Learning/runs/Mar31_11-53-50_M3009R21854/FINALPolicy.pth')


done = False
s = env.reset()
s = env.reset()


env.render()
R = []

while not done:

    a = multiagent.select_action(s)
    s,r,done,i = env.step(a)
    print(np.sum(i['individual_rewards']))
    R.append(i['individual_rewards'])
    env.render()

plt.show()
plt.close()
print(np.sum(R))
R = np.asarray(R)
plt.plot(np.cumsum(R, axis=0))
plt.show()
