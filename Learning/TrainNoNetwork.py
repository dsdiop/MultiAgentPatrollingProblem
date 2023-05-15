import sys
import os
data_path = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(data_path)
from Environment.PatrollingEnvironments import MultiAgentPatrolling
from Algorithm.RainbowDQL.Agent.DuelingDQNAgent import MultiAgentDuelingDQNAgent
import numpy as np
import torch

N = 4
sc_map = np.genfromtxt(data_path+'/Environment/Maps/example_map.csv', delimiter=',')
visitable_locations = np.vstack(np.where(sc_map != 0)).T
random_index = np.random.choice(np.arange(0,len(visitable_locations)), N, replace=False)
initial_positions = np.asarray([[24, 21],[28,24],[27,19],[24,24]])

#initial_positions = np.array([[30, 20], [32, 20], [34, 20], [30, 22]])[:N, :]
#visitable = np.column_stack(np.where(sc_map == 1))
# initial_positions = visitable[np.random.randint(0,len(visitable), size=N), :]

#frame_stack
nettype = '0'

weight = ['nashmtl',None, None]
for i,ww in enumerate(weight):
    env = MultiAgentPatrolling(scenario_map=sc_map,
                            fleet_initial_positions=initial_positions,
                            distance_budget=200,
                            number_of_vehicles=N,
                            seed=0,
                            miopic=True,
                            detection_length=2,
                            movement_length=2,
                            max_collisions=15,
                            forget_factor=0.5,
                            attrition=0.1,
                            networked_agents=False,
                            ground_truth_type='algae_bloom',
                            obstacles=False,
                            frame_stacking=1,
                            state_index_stacking=(2, 3, 4),
                            reward_weights=(1.0, 0.1)
                            )
nuin = [[0., 1], [0.30, 1], [0.60, 0.], [1., 0.]]
if i==0:
    cc = 'nashmtl'
elif i==1:
    cc = 'infclipped'
elif i==2:
    nuin = [[0., 1], [0.30, 1], [0.30, 0.], [1., 0.]]
    cc = 'escalon'
    
    multiagent = MultiAgentDuelingDQNAgent(env=env,
                                        memory_size=int(1E6),
                                        batch_size=64,#64
                                        target_update=1000,
                                        soft_update=True,
                                        tau=0.001,
                                        epsilon_values=[1.0, 0.05],
                                        epsilon_interval=[0.0, 0.5],
                                        learning_starts=100, # 100
                                        gamma=0.99,
                                        lr=1e-4,
                                        number_of_features=1024,
                                        noisy=False,
                                        nettype='0',
                                        archtype='v1',
                                        weighted=False,
                                        train_every=15,
                                        save_every=1000,
                                        distributional=False,
                                        logdir=f'Learning/runs/Vehicles_{N}/Experimento_serv_14_'+'_net_'+nettype+'_'+cc,
                                        use_nu=True,
                                        nu_intervals=nuin,
                                        eval_episodes=10,
                                        eval_every=1000,
                                        use_dwa=False,
                                        weighting_method=ww,
                                        weight_methods_parameters=dict(
                                        update_weights_every=2,
                                        optim_niter=20)
                                        )

    multiagent.train(episodes=20000)
    torch.cuda.empty_cache()
