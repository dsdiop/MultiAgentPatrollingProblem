
import sys
import os
data_path = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(data_path)
from Environment.PatrollingEnvironments import MultiAgentPatrolling
from Algorithm.RainbowDQL.Agent.DuelingDQNAgent import MultiAgentDuelingDQNAgent
import numpy as np
import matplotlib.pyplot as plt
import torch
from Utils.metrics_wrapper import MetricsDataCreator
import json
from tqdm import trange
import pandas as pd 
import seaborn as sns 

def EvaluateMultiagent(number_of_agents: int,
                       sc_map,
                       visitable_locations,
                       initial_positions,
                       num_of_eval_episodes: int,
                       policy_path: str,
                       policy_type: str,
                       seed: int,
                       policy_name: str='DDQN',
                       metrics_directory: str= './',
                       agent_config=None,
                       environment_config=None,
                       render=False
                       ):

    N = number_of_agents
    random_index = np.random.choice(np.arange(0,len(visitable_locations)), N, replace=False)

    
    if agent_config is None:
        agent_config = json.load(open(f'{policy_path}experiment_config.json', 'rb'))
    
    if environment_config is None:
        environment_config = json.load(open(f'{policy_path}environment_config.json', 'rb'))

    ## Some Sanity checks
    try:
        environment_config['movement_length']=environment_config['movement_length']
    except:
        environment_config['movement_length'] = 2
        
    try:
        environment_config['frame_stacking']=environment_config['frame_stacking']
        environment_config['state_index_stacking']=environment_config['state_index_stacking']
    except:
        if 'fstack' in policy_path:
            environment_config['frame_stacking'] = 2
        else:
            environment_config['frame_stacking'] = 1
        
        environment_config['state_index_stacking']=(2, 3, 4)
        
    env = MultiAgentPatrolling(scenario_map=sc_map,
                                fleet_initial_positions=environment_config['fleet_initial_positions'],
                                distance_budget=environment_config['distance_budget'],
                                number_of_vehicles=environment_config['number_of_agents'],
                                seed=seed,
                                miopic=environment_config['miopic'],
                                detection_length=environment_config['detection_length'],
                                movement_length=environment_config['movement_length'],
                                max_collisions=environment_config['max_number_of_colissions'],
                                forget_factor=environment_config['forgetting_factor'],
                                attrition=environment_config['attrition'],
                                networked_agents=False,
                                ground_truth_type=environment_config['ground_truth'],
                                obstacles=False,
                                frame_stacking=environment_config['frame_stacking'],
                                state_index_stacking=environment_config['state_index_stacking'],
                                reward_weights=environment_config['reward_weights']
                                )

    ## Some Sanity checks
    try:
        agent_config['nettype']=agent_config['nettype']
    except:
        agent_config['nettype'] = '0'
        
    try:
        agent_config['archtype']=agent_config['archtype']
    except:
        if 'v2' in policy_path:
            agent_config['archtype'] = 'v2'
        else:
            agent_config['archtype'] = 'v1'
        
    multiagent = MultiAgentDuelingDQNAgent(env=env,
                                        memory_size=int(1),
                                        batch_size=64,
                                        target_update=1000,
                                        soft_update=True,
                                        tau=0.001,
                                        epsilon_values=[0, 0],
                                        epsilon_interval=[0.0, 0.5],
                                        learning_starts=100, # 100
                                        gamma=0.99,
                                        lr=1e-4,
                                        number_of_features=1024 if 'n_of_features_512' not in policy_path else 512,
                                        noisy=False,
                                        nettype=agent_config['nettype'],
                                        archtype=agent_config['archtype'],
                                        train_every=15,
                                        save_every=1000,
                                        distributional=False,
                                        logdir=f'Learning/runs/Vehicles_{N}/{policy_path}',
                                        use_nu=agent_config['use_nu'],
                                        nu_intervals=agent_config['nu_intervals'],
                                        eval_episodes=num_of_eval_episodes,
                                        eval_every=1000)

    multiagent.load_model(policy_path+policy_type)
    metrics = MetricsDataCreator(metrics_names=['Policy Name',
                                                'Accumulated Reward Intensification',
                                                'Accumulated Reward Exploration',
                                                'Total Accumulated Reward',
                                                'Total Length',
                                                'Total Collisions',
                                                'Nu'],
                                algorithm_name='DRL',
                                experiment_name='DRLResults',
                                directory=metrics_directory)

    if os.path.exists(metrics_directory + 'DRLResults' + '.csv'):
        metrics.load_df(metrics_directory + 'DRLResults' + '.csv')
        
    paths = MetricsDataCreator(metrics_names=['vehicle', 'x', 'y'],
                            algorithm_name='DRL',
                            experiment_name='DRL_paths',
                            directory=metrics_directory)
    
    if os.path.exists(metrics_directory + 'DRL_paths' + '.csv'):
        paths.load_df(metrics_directory + 'DRL_paths' + '.csv')
        
    """ Evaluate the agent on the environment for a given number of episodes with a deterministic policy """

    multiagent.dqn.eval()
    max_movements = env.distance_budget
    multiagent.epsilon = 0
    
    
    for run in trange(num_of_eval_episodes):

        # Reset the environment #
        state = env.reset()
        recompensa_exp = []
        recompensa_inf = []
        if render and run==0:
            env.render()
        done = {agent_id: False for agent_id in range(env.number_of_agents)}

        total_reward = 0
        total_reward_information = 0
        total_reward_exploration = 0
        total_length = 0
        total_collisions = 0
        
        metrics_list = [policy_name, total_reward_information, total_reward_exploration, total_reward, total_length, total_collisions, multiagent.nu]
        # Initial register #
        metrics.register_step(run_num=run, step=total_length, metrics=metrics_list)
        for veh_id, veh in enumerate(env.fleet.vehicles):
            paths.register_step(run_num=run, step=total_length, metrics=[veh_id, veh.position[0], veh.position[1]])
        
        while not all(done.values()):

            total_length += 1
            if multiagent.use_nu:
                distance = np.min([np.max(env.fleet.get_distances()), max_movements])
                multiagent.nu = multiagent.anneal_nu(p= distance / max_movements,
                                        p1=multiagent.nu_intervals[0],
                                        p2=multiagent.nu_intervals[1],
                                        p3=multiagent.nu_intervals[2],
                                        p4=multiagent.nu_intervals[3])
                
                #print(multiagent.nu)
            # Select the action using the current policy
            
            if not multiagent.masked_actions:
                actions = multiagent.select_action(state)
            else:
                actions = multiagent.select_masked_action(states=state, positions=env.fleet.get_positions())


            actions = {agent_id: action for agent_id, action in actions.items() if not done[agent_id]}

            # Process the agent step #
            next_state, reward, done = multiagent.step(actions)
            #print(reward)
            if render and run==0:
                env.render()

            # Update the state #
            state = next_state
            rewards = np.asarray(list(reward.values()))
            total_reward_information += np.sum(rewards[:,0])
            total_reward_exploration += np.sum(rewards[:,1])
            
            recompensa_exp.append(total_reward_exploration)
            recompensa_inf.append(total_reward_information)
            total_collisions += env.fleet.fleet_collisions    
            total_reward = total_reward_exploration + total_reward_information
            metrics_list = [policy_name, total_reward_information, total_reward_exploration, total_reward, total_length, total_collisions, multiagent.nu]
            metrics.register_step(run_num=run, step=total_length, metrics=metrics_list)
            for veh_id, veh in enumerate(env.fleet.vehicles):
                paths.register_step(run_num=run, step=total_length, metrics=[veh_id, veh.position[0], veh.position[1]])
        """fig, ax = plt.subplots()
        ax.plot(recompensa_exp, label='Exploration reward')
        ax.legend()
        ax.plot(recompensa_inf, label='Information reward')
        ax.legend()  
        plt.show()  """     
    metrics.register_experiment()
    paths.register_experiment()

    # Return the average reward, average length
    mean_reward_inf = total_reward_information / num_of_eval_episodes
    mean_reward_exp = total_reward_exploration / num_of_eval_episodes
    mean_reward = total_reward / num_of_eval_episodes
    mean_length = total_length / num_of_eval_episodes
    mean_collisions = total_collisions/ num_of_eval_episodes

    return mean_reward_inf, mean_reward_exp, mean_reward, mean_length, mean_collisions



if __name__ == '__main__':
    if False:
        N = 4
        sc_map = np.genfromtxt('Environment/Maps/example_map.csv', delimiter=',')
        visitable_locations = np.vstack(np.where(sc_map != 0)).T
        random_index = np.random.choice(np.arange(0,len(visitable_locations)), N, replace=False)
        initial_positions = np.asarray([[24, 21],[28,24],[27,19],[24,24]])
        num_of_eval_episodes = 100
        policy_names = [ 'Experimento_serv_0_nettype_0',
                        'Experimento_serv_0_nettype_1',
                        'Experimento_serv_1_lr2_nettype_0',
                        'Experimento_serv_2_nettype_0',
                        'Experimento_serv_3_bs64_nettype_0',
                        'Experimento_serv_3_bs256_nettype_0',
                        'Experimento_serv_4_rewardv2_nettype_0',
                        'Experimento_serv_5_fstack2_nettype_0',
                        'Experimento_serv_6_bs64_nettype_2',
                        'Experimento_serv_7_bs64_nettype_3',
                        'Experimento_serv_7_bs64_nettype_4',
                        'Experimento_serv_8_nettype_0_n_of_features_1024_archtype_v2',
                        'Experimento_serv_8_nettype_5_n_of_features_512_archtype_v1',
                        'Experimento_serv_9_nettype_0_archtype_v1',
                        'Experimento_serv_9_nettype_0_archtype_v2',
                        'Experimento_serv_10_nettype_0_archtype_v1',
                        'Experimento_serv_10_nettype_0_archtype_v2',
                        'Experimento_serv_11_nettype_0_archtype_v1',
                        'Experimento_serv_11_nettype_0_archtype_v2',
                        'Experimento_serv_13__v1_wei_True_gt_algae_bloom_db_200_i1',
                        'Experimento_serv_13__v1_wei_False_gt_algae_bloom_db_400_i2',
                        'Experimento_serv_13__v1_wei_False_gt_algae_bloom_db_200_i3',
                        'Experimento_serv_13__v2_wei_False_gt_algae_bloom_db_200_i4',
                        'Experimento_serv_13__v1_wei_False_gt_shekel_db_200_i5',
                        'Experimento_serv_14__net_0_nashmtl',
                        'Experimento_serv_14__net_0_nashmtl_100',
                        'Experimento_serv_14__net_0_pcgrad', 
                        'Experimento_serv_14__net_0_rlw',
                        'Experimento_serv_14__net_0_scaleinvls',
                        'Experimento_serv_14__net_0_cagrad',
                        'Experimento_serv_14__net_0_escalon',
                        'Experimento_serv_14__net_0_imtl',
                        'Experimento_serv_14__net_0_infclipped',
                        'Experimento_serv_14__net_0_mgda']
        for i,policy_name in enumerate(policy_names):
            if 'Experimento_serv_7_bs64_nettype_4' not in policy_name:
                continue
            print(policy_name)
            policy_path = f'../DameLearnings/runs/Vehicles_4/{policy_name}/'
            policy_type = 'Final_Policy.pth'
            seed = 17
            EvaluateMultiagent(number_of_agents=N,
                            sc_map=sc_map,
                            visitable_locations=visitable_locations,
                            initial_positions=initial_positions,
                            num_of_eval_episodes=num_of_eval_episodes,
                            policy_path=policy_path,
                            policy_type=policy_type,
                            seed=seed,
                            policy_name=policy_name,
                            render = True
                            )
    Finalpolicy = pd.read_csv(f'{data_path}/Evaluation/Results/BestPolicy_reward_explorationDRLResults.csv')
    Accum_per_episode = Finalpolicy.groupby(['Policy Name','Run'])[['Accumulated Reward Intensification', 'Accumulated Reward Exploration']].tail(1) 
    #print(Accum_per_episode.to_markdown(),'\n \n \n')
    # merge the result dataframe with the original dataframe on the 'group' and 'value' columns
    Finalpolicy_accum = Finalpolicy.loc[Accum_per_episode.index]
    #print(Finalpolicy_accum.to_markdown(),'\n \n \n')
    Mean_per_episode = Finalpolicy_accum.groupby('Policy Name')[['Accumulated Reward Intensification', 'Accumulated Reward Exploration']].mean()
    print(Mean_per_episode.sort_values('Accumulated Reward Exploration',ascending=False).to_markdown(),'\n \n \n')
    from pymoo.factory import get_performance_indicator
    from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

    # extract the two objective values from the dataframe into a numpy array
    objs = - Mean_per_episode[['Accumulated Reward Intensification', 'Accumulated Reward Exploration']].to_numpy()


    nds = NonDominatedSorting()
    fronts = nds.do(objs)

    # get the solutions in the first front (i.e., the Pareto front)
    pf = objs[fronts[0]]

    # print the Pareto front
    print("Pareto front:")
    print(pf)

    
    sns.relplot(
    data=Mean_per_episode.sort_values('Accumulated Reward Exploration',ascending=False), kind="scatter",
    x="Accumulated Reward Exploration", y="Accumulated Reward Intensification", hue="Policy Name"
)
    plt.show()