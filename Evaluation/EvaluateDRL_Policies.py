
import sys
import os
data_path = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(data_path)
from Environment.PatrollingEnvironments import MultiAgentPatrolling
from Algorithm.RainbowDQL.Agent.DuelingDQNAgent import MultiAgentDuelingDQNAgent
from Environment.GroundTruthsModels.AlgaeBloomGroundTruth import algae_colormap,background_colormap
import numpy as np
import matplotlib.pyplot as plt
import torch
from Utils.metrics_wrapper import MetricsDataCreator
import json
from tqdm import trange
import pandas as pd 
import seaborn as sns 
#from Evaluation.Utils.path_plotter import plot_trajectory
from Algorithm.RainbowDQL.Agent.pruebas import plot_visits, plot_state
from Evaluation.Utils.EvaluationUtils import run_path_planners_evaluation
imm = []
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
                       nu_interval = None,
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
                                #networked_agents=False,
                                ground_truth_type=environment_config['ground_truth'],
                                obstacles=False,
                                frame_stacking=environment_config['frame_stacking'],
                                state_index_stacking=environment_config['state_index_stacking'],
				                reward_type='metrics global',
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
                                        device='cuda',
                                        train_every=15,
                                        save_every=1000,
                                        distributional=False,
                                        logdir=f'Learning/runs/Vehicles_{N}/{policy_path}',
                                        use_nu=agent_config['use_nu'],
                                        nu_intervals=agent_config['nu_intervals'] if nu_interval is None else nu_interval,
                                        eval_episodes=num_of_eval_episodes,
                                        eval_every=1000)

    multiagent.load_model(policy_path+policy_type)
    metrics = MetricsDataCreator(metrics_names=['Policy Name',
                                                'Accumulated Reward Intensification',
                                                'Accumulated Reward Exploration',
                                                'Total Accumulated Reward',
                                                'Total Length',
                                                'Total Collisions',
                                                'Average global idleness Intensification',
                                                'Average global idleness Exploration',
                                                'Sum global idleness Intensification',
                                                'Percentage Visited Exploration',
                                                'Percentage Visited'],
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
        percentage_visited = np.count_nonzero(env.fleet.historic_visited_mask) / np.count_nonzero(env.scenario_map)
        percentage_visited_exp = np.count_nonzero(env.fleet.historic_visited_mask) / np.count_nonzero(env.scenario_map)
        average_global_idleness_exp = env.average_global_idleness_exp
        sum_global_interest = env.sum_global_idleness
        sum_instantaneous_global_idleness = 0
        steps_int = 0
        average_global_idleness_int = sum_instantaneous_global_idleness
        metrics_list = [policy_name, total_reward_information,
                        total_reward_exploration,
                        total_reward, total_length,
                        total_collisions,
                        average_global_idleness_int,
                        average_global_idleness_exp,
                        sum_global_interest,
                        percentage_visited_exp,
                        percentage_visited]
        # Initial register #
        metrics.register_step(run_num=run, step=total_length, metrics=metrics_list)
        for veh_id, veh in enumerate(env.fleet.vehicles):
            paths.register_step(run_num=run, step=total_length, metrics=[veh_id, veh.position[0], veh.position[1]])
        dd = True
        #fig,axs = plt.subplots(1, 4, figsize=(15,5))
        fig_vis = []
        while not all(done.values()):

            total_length += 1
            if multiagent.use_nu:
                distance = np.min([np.max(env.fleet.get_distances()), max_movements])
                multiagent.nu = multiagent.anneal_nu(p= distance / max_movements,
                                        p1=multiagent.nu_intervals[0],
                                        p2=multiagent.nu_intervals[1],
                                        p3=multiagent.nu_intervals[2],
                                        p4=multiagent.nu_intervals[3])
                if render:
                    print(multiagent.nu)
            # Select the action using the current policy
            
            if not  multiagent.masked_actions:
                actions = multiagent.select_action(state)
            else:
                actions = multiagent.select_masked_action(states=state, positions=env.fleet.get_positions())
         

            actions = {agent_id: action for agent_id, action in actions.items() if not done[agent_id]}
            #print(env.fleet.get_positions())
            # Process the agent step #
            next_state, reward, done = multiagent.step(actions)
            
            if total_length==40:
                fsagefsdvefsgd=0
                #print(hey)
                #env.node_visit=np.zeros_like(env.scenario_map)
            if multiagent.nu < 0.5 and dd and False:
                
                dd = False
                fig0,ax = plt.subplots()
                pos0= ax.imshow(env.node_visit, cmap='rainbow')
                fig0.colorbar(pos0, ax=ax, orientation='vertical')
                #plt.show()
                fig_vis.append(env.node_visit)
                #plt.savefig(f'C:\\Users\\dames\\OneDrive\\Documentos\\GitHub\\MultiAgentPatrollingProblem\\Results_seed30_firstpaper1/{policy_name}_node_visit_exp.png')
                plt.close()
                #print('exp: ', percentage_visited_exp)
                fig1,ax = plt.subplots()
                model = env.scenario_map*np.nan
                model[np.where(env.scenario_map)] = env.model[np.where(env.scenario_map)]
                pos1 = ax.imshow(model,  cmap=algae_colormap, vmin=0.0, vmax=1.0)
                fig1.colorbar(pos1, ax=ax, orientation='vertical')
                fig_vis.append(model)   
                plt.title('Contamination Model')
                #plt.savefig(f'C:\\Users\\dames\\OneDrive\\Documentos\\GitHub\\MultiAgentPatrollingProblem\\Results_seed30_firstpaper1/contamination_model.png')
                #plt.show()
                plt.close()
                
                env.node_visit=np.zeros_like(env.scenario_map)
            #print(reward)
            if render and run==0:
                env.render()
                """
                axs[0].imshow(env.im4.get_array(),cmap=algae_colormap,vmin=0,vmax=1)
                axs[0].set_title("Real importance GT")
                axs[1].imshow(env.im1.get_array(),cmap = 'rainbow_r')
                axs[1].set_title('Idleness map (W)')
                axs[2].imshow(env.im3.get_array(),cmap=algae_colormap,vmin=0,vmax=1)
                axs[2].set_title("Intantaneous Model / Importance (I)")
                axs[3].imshow(env.im7.get_array(),vmin=0,vmax=4)
                axs[3].set_title("Redundacy Mask")
                #plt.colorbar(im,ax=ax)"""
            
            #print(reward)
            reward_idl0 = [rew[0]/env.instantaneous_global_idleness_exp for id,rew in reward.items()]
            reward_idl = [rew[1]/env.instantaneous_global_idleness_exp for id,rew in reward.items()]
            #print(f'global idleness information is : {env.instantaneous_global_idleness} rewdiv: {reward_idl0}')
            #print(f'global idleness exploration is : {env.instantaneous_global_idleness_exp} rewdiv: {reward_idl}')
            # Update the state #
            state = next_state
            rewards = np.asarray(list(reward.values()))
            percentage_visited = np.count_nonzero(env.fleet.historic_visited_mask) / np.count_nonzero(env.scenario_map)
            if multiagent.nu<0.5:
                steps_int += 1
                sum_instantaneous_global_idleness += env.instantaneous_global_idleness
                average_global_idleness_int = sum_instantaneous_global_idleness/steps_int
                total_reward_information += np.sum(rewards[:,0])
            else:
                average_global_idleness_exp = np.copy(env.average_global_idleness_exp)
                percentage_visited_exp = np.count_nonzero(env.fleet.historic_visited_mask) / np.count_nonzero(env.scenario_map)
                total_reward_exploration += np.sum(rewards[:,1])
            
            recompensa_exp.append(total_reward_exploration)
            recompensa_inf.append(total_reward_information)
            total_collisions += env.fleet.fleet_collisions    
            total_reward = total_reward_exploration + total_reward_information
            
            
            sum_global_interest = env.sum_global_idleness
            metrics_list = [policy_name, total_reward_information,
                            total_reward_exploration,
                            total_reward, total_length,
                            total_collisions,
                            average_global_idleness_int,
                            average_global_idleness_exp,
                            sum_global_interest,
                            percentage_visited_exp,
                            percentage_visited]
            metrics.register_step(run_num=run, step=total_length, metrics=metrics_list)
            for veh_id, veh in enumerate(env.fleet.vehicles):
                paths.register_step(run_num=run, step=total_length, metrics=[veh_id, veh.position[0], veh.position[1]])
        """fig, ax = plt.subplots()
        ax.plot(recompensa_exp, label='Exploration reward')
        ax.legend()
        ax.plot(recompensa_inf, label='Information reward')
        ax.legend()  
        plt.show()  """ 
    if dd == False: 
        fig2,ax = plt.subplots()
        pos2 = ax.imshow(env.node_visit, cmap='rainbow')
        fig2.colorbar(pos2, ax=ax, orientation='vertical')
        fig_vis.append(env.node_visit)
        #plt.savefig(f'C:\\Users\\dames\\OneDrive\\Documentos\\GitHub\\MultiAgentPatrollingProblem\\Results_seed30_firstpaper1/{policy_name}_node_visit_int.png')
        #plt.show()
        plt.close()
        #print('int:', percentage_visited)
        imm.append({policy_name:fig_vis})
        json.dump(dict(policy_name=policy_name,
                       total_reward_information=float(total_reward_information),
                            total_reward_exploration=float(total_reward_exploration),
                            total_reward=float(total_reward),
                            total_length=float(total_length),
                            total_collisions=float(total_collisions),
                            average_global_idleness_int=float(average_global_idleness_int),
                            average_global_idleness_exp=float(average_global_idleness_exp),
                            sum_global_interest=float(sum_global_interest),
                            percentage_visited_exp=float(percentage_visited_exp),
                            percentage_visited=float(percentage_visited)),
                        open(f'C:\\Users\\dames\\OneDrive\\Documentos\\GitHub\\MultiAgentPatrollingProblem\\Results_seed30_firstpaper1/imagenes_visitas/{policy_name}_seed_{seed}.json', 'w'), indent=4)
    if not render:
        metrics.register_experiment()
        paths.register_experiment()
    else:
        plt.close()
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
        sc_map = np.genfromtxt(f'{data_path}/Environment/Maps/example_map.csv', delimiter=',')
        visitable_locations = np.vstack(np.where(sc_map != 0)).T
        random_index = np.random.choice(np.arange(0,len(visitable_locations)), N, replace=False)
        initial_positions = np.asarray([[24, 21],[28,24],[27,19],[24,24]])
        num_of_eval_episodes = 200
        policy_names_veh4 = [ #'Experimento_serv_0_nettype_0',
                        #'Experimento_serv_0_nettype_1',
                        #'Experimento_serv_1_lr2_nettype_0',
                        #'Experimento_serv_2_nettype_0',
                        #'Experimento_serv_3_bs64_nettype_0',
                        #'Experimento_serv_3_bs256_nettype_0',
                        #'Experimento_serv_4_rewardv2_nettype_0',
                        #'Experimento_serv_5_fstack2_nettype_0',
                        'Experimento_serv_7_bs64_nettype_3',
                        'Experimento_serv_7_bs64_nettype_4',
                        'Experimento_serv_8_nettype_0_n_of_features_1024_archtype_v2',
                        #'Experimento_serv_8_nettype_5_n_of_features_512_archtype_v1',
                        #'Experimento_serv_9_nettype_0_archtype_v1',
                        #'Experimento_serv_9_nettype_0_archtype_v2',
                        #'Experimento_serv_10_nettype_0_archtype_v1',
                        'Experimento_serv_10_nettype_0_archtype_v2',
                        #'Experimento_serv_11_nettype_0_archtype_v1',
                        #'Experimento_serv_11_nettype_0_archtype_v2',
                        #'Experimento_serv_13__v1_wei_True_gt_algae_bloom_db_200_i1',
                        #'Experimento_serv_13__v1_wei_False_gt_algae_bloom_db_400_i2',
                        #'Experimento_serv_13__v1_wei_False_gt_algae_bloom_db_200_i3',
                        #'Experimento_serv_13__v2_wei_False_gt_algae_bloom_db_200_i4',
                        #'Experimento_serv_13__v1_wei_False_gt_shekel_db_200_i5',
                        #'Experimento_serv_14__net_0_nashmtl',
                        'Experimento_serv_14__net_0_nashmtl_100',
                        'Experimento_serv_14__net_0_pcgrad', 
                        #'Experimento_serv_14__net_0_rlw',
                        #'Experimento_serv_14__net_0_scaleinvls',
                        #'Experimento_serv_14__net_0_cagrad',
                        #'Experimento_serv_14__net_0_escalon',
                        'Experimento_serv_14__net_0_imtl',
                        #'Experimento_serv_14__net_0_infclipped',
                        'Experimento_serv_14__net_0_mgda',
                        'Experimento_serv_15__net_0___v2',
                        'Experimento_serv_15__net_0__x3c__v1',
                        'Experimento_serv_15__net_0_nashmtl_v1',
                        'Experimento_serv_15__net_0_nashmtl_1000',
                        'Experimento_serv_17_rewardv3_net_0v1',
                        'Experimento_serv_18__net_4v1notmasked',
                        'Experimento_serv_19__net_4_arch_v1_rewv1',
                        'Experimento_serv_19__net_4_arch_v1_rewv2',
                        'Experimento_serv_19__net_4_arch_v2_rewv1',
                        'Experimento_serv_19__net_4_arch_v2_rewv2',
                        'Experimento_serv_20__net_0_arch_v1_rewv2_notmasked_detlength_1',
                        'Experimento_serv_21__net_0_arch_v1_rewv2_minimumidleness',
                        'Experimento_serv_22__net_0_arch_v1_rewv2_minimumidleness_menos2',
                        'Experimento_serv_22__net_0_arch_v2_rewv2_minimumidleness_menos2',
                        'Experimento_serv_22__net_0_arch_v1_rewv2_minimumidleness_menos1',
                        'Experimento_serv_22__net_0_arch_v2_rewv2_minimumidleness_menos1',
                        'Experimento_serv_23__net_0_arch_v1_rewv2_v2v3']
        policy_names_veh4 = ['Experimento_serv_22__net_0_arch_v1_rewv2_no_cost',
                        'Experimento_serv_22__net_0_arch_v2_rewv2_no_cost',
                        'Experimento_serv_24_net_4_arch_v1_rew_v2',
                        'Experimento_serv_24_net_4_arch_v2_rew_v2',
                        'Experimento_serv_25_net_0_arch_v1_rew_v2_weight_dwa',
                        'Experimento_serv_25_net_0_arch_v1_rew_v2_weight_imtl',
                        'Experimento_serv_25_net_0_arch_v1_rew_v2_weight_mgda',
                        'Experimento_serv_25_net_0_arch_v1_rew_v2_weight_nashmtl',
                        'Experimento_serv_25_net_0_arch_v1_rew_v2_weight_scaleinvls',
                        'Experimento_serv_25_net_0_arch_v2_rew_v2_weight_dwa',
                        'Experimento_serv_25_net_0_arch_v2_rew_v2_weight_imtl',
                        'Experimento_serv_25_net_0_arch_v2_rew_v2_weight_mgda',
                        'Experimento_serv_25_net_0_arch_v2_rew_v2_weight_nashmtl',
                        'Experimento_serv_25_net_0_arch_v2_rew_v2_weight_scaleinvls',
                        'Experimento_serv_25_net_4_arch_v1_rew_v2_weight_dwa',
                        'Experimento_serv_25_net_4_arch_v1_rew_v2_weight_imtl',
                        'Experimento_serv_25_net_4_arch_v1_rew_v2_weight_mgda',
                        'Experimento_serv_25_net_4_arch_v1_rew_v2_weight_nashmtl',
                        'Experimento_serv_25_net_4_arch_v1_rew_v2_weight_scaleinvls',
                        'Experimento_serv_25_net_4_arch_v2_rew_v2_weight_dwa',
                        'Experimento_serv_25_net_4_arch_v2_rew_v2_weight_imtl',
                        'Experimento_serv_25_net_4_arch_v2_rew_v2_weight_mgda',
                        'Experimento_serv_25_net_4_arch_v2_rew_v2_weight_nashmtl',
                        'Experimento_serv_25_net_4_arch_v2_rew_v2_weight_scaleinvls',
                        'Experimento_serv_26_net_0_arch_v1_idleness_zero_out',
                        'Experimento_serv_26_net_4_arch_v1_idleness_zero_out',
                        'Experimento_serv_26_net_0_arch_v2_idleness_zero_out',
                        'Experimento_serv_26_net_4_arch_v2_idleness_zero_out',
                        'Experimento_serv_27__net_0_arch_v2_rewv2_v3_IGI_div',
                        'Experimento_serv_27__net_0_arch_v2_rewv4_WLU',
                        'Experimento_serv_27__net_0_arch_v1_rewv2_v3_IGI_div',
                        'Experimento_serv_27__net_0_arch_v1_rewv4_WLU',
                        'Experimento_serv_27__net_4_arch_v2_rewv2_v3_IGI_div',
                        'Experimento_serv_27__net_4_arch_v2_rewv4_WLU',
                        'Experimento_serv_27__net_4_arch_v1_rewv2_v3_IGI_div',
                        'Experimento_serv_27__net_4_arch_v1_rewv4_WLU']
        policy_names_veh4 = ['Experimento_serv_22__net_0_arch_v1_rewv2_no_cost',
                        'Experimento_serv_22__net_0_arch_v2_rewv2_no_cost',
                        'Experimento_serv_24_net_4_arch_v1_rew_v2',
                        'Experimento_serv_24_net_4_arch_v2_rew_v2',
                        'Experimento_serv_27__net_0_arch_v1_rewv4_WLU',
                        'Experimento_serv_27__net_0_arch_v2_rewv4_WLU',
                        'Experimento_serv_27__net_4_arch_v1_rewv4_WLU',
                        'Experimento_serv_27__net_4_arch_v2_rewv4_WLU']
        policy_names_veh4_first_paper = ['Experimento_serv_2_net_0_arch_v1_rew_v4',
                                         'Experimento_serv_2_net_0_arch_v1_rew_v4_one_agent_saves_in_buffer',
                                         'Experimento_serv_2_net_0_arch_v2_rew_v4',
                                         'Experimento_serv_2_net_0_arch_v1_rew_v4_one_agent_saves_in_buffer_first_one',
                                         'Experimento_serv_2_net_0_arch_v2_rew_v4_one_agent_saves_in_buffer',
                                         'Experimento_serv_2_net_4_arch_v1_rew_v4',
                                         'Experimento_serv_2_net_4_arch_v1_rew_v4_one_agent_saves_in_buffer',
                                         'Experimento_serv_2_net_4_arch_v2_rew_v4',
                                         'Experimento_serv_2_net_4_arch_v2_rew_v4_one_agent_saves_in_buffer',
                                         'Experimento_serv_2_net_0_arch_v1_rew_v2',
                                         'Experimento_serv_2_net_0_arch_v1_rew_v2_one_agent_saves_in_buffer',
                                         'Experimento_serv_2_net_0_arch_v2_rew_v2',
                                         'Experimento_serv_2_net_0_arch_v2_rew_v2_one_agent_saves_in_buffer',
                                         'Experimento_serv_2_net_4_arch_v1_rew_v2',
                                         'Experimento_serv_2_net_4_arch_v1_rew_v2_one_agent_saves_in_buffer',
                                         'Experimento_serv_2_net_4_arch_v2_rew_v2',
                                         'Experimento_serv_2_net_4_arch_v2_rew_v2_one_agent_saves_in_buffer']
        policy_names_veh4_first_paper0 = ['Experimento_serv_3_net_0_arch_v1_rew_v2_weight_in_PER_imtl',
                                         'Experimento_serv_3_net_0_arch_v1_rew_v2_weight_in_PER_imtl_one_agent_saves_in_buffer',
                                         'Experimento_serv_3_net_0_arch_v1_rew_v4_weight_in_PER_imtl',
                                         'Experimento_serv_3_net_0_arch_v1_rew_v4_weight_in_PER_imtl_one_agent_saves_in_buffer',
                                         'Experimento_serv_3_net_0_arch_v1_rew_v2_weight_in_PER_nashmtl',
                                         'Experimento_serv_3_net_0_arch_v1_rew_v2_weight_in_PER_nashmtl_one_agent_saves_in_buffer',
                                         'Experimento_serv_3_net_0_arch_v1_rew_v2_alphaPER_06_beta_04',
                                         'Experimento_serv_3_net_0_arch_v1_rew_v2_n_steps_5',
                                         'Experimento_serv_3_net_0_arch_v1_rew_v4_alphaPER_06_beta_04',
                                         'Experimento_serv_3_net_0_arch_v1_rew_v2_n_steps_10',
                                         'Experimento_serv_3_net_0_arch_v2_rew_v2_alphaPER_06_beta_04',
                                         'Experimento_serv_3_net_0_arch_v1_rew_v2_n_steps_20',
                                         'Experimento_serv_4_net_0_arch_v1_rew_v5_alpha_0.2_n_steps_5',
                                         'Experimento_serv_4_net_0_arch_v1_rew_v5_alpha_0.6_n_steps_5',
                                         'Experimento_serv_4_net_0_arch_v2_rew_v5_alpha_0.2_n_steps_5',
                                         'Experimento_serv_4_net_0_arch_v2_rew_v5_alpha_0.6_n_steps_5',
                                         'Experimento_serv_4_net_0_arch_v1_rew_v2_alpha_0.4_n_steps_5',
                                         'Experimento_serv_4_net_0_arch_v2_rew_v2_alpha_0.4_n_steps_5',
                                         'Experimento_serv_4_net_0_arch_v1_rew_v2_alpha_0.5_n_steps_5',
                                         'Experimento_serv_4_net_0_arch_v2_rew_v2_alpha_0.5_n_steps_5',
                                         'Experimento_serv_4_net_0_arch_v1_rew_v5_alpha_0.2_n_steps_1',
                                         'Experimento_serv_4_net_0_arch_v1_rew_v5_alpha_0.6_n_steps_1',
                                         'Experimento_serv_4_net_0_arch_v2_rew_v5_alpha_0.2_n_steps_1',
                                         'Experimento_serv_4_net_0_arch_v2_rew_v5_alpha_0.6_n_steps_1',
                                         'Experimento_serv_4_net_0_arch_v1_rew_v2_alpha_0.4_n_steps_1',
                                         'Experimento_serv_4_net_0_arch_v2_rew_v2_alpha_0.4_n_steps_1',
                                         'Experimento_serv_4_net_0_arch_v1_rew_v2_alpha_0.5_n_steps_1',
                                         'Experimento_serv_4_net_0_arch_v2_rew_v2_alpha_0.5_n_steps_1',
                                         'Experimento_serv_5_net_0_arch_v2_rew_v4_weight_imtl_n_steps_5',
                                         'Experimento_serv_5_net_0_arch_v2_rew_v2_weight_imtl_n_steps_5',
                                         'Experimento_serv_5_net_0_arch_v1_rew_v4_weight_imtl_n_steps_5',
                                         'Experimento_serv_5_net_0_arch_v1_rew_v2_weight_imtl_n_steps_5']
        policy_names_veh2 =['Experimento_serv_16__net_0v1','Experimento_serv_19__net_4_arch_v1_rewv1']
        policy_names_veh3 =['Experimento_serv_16__net_0v1','Experimento_serv_19__net_4_arch_v1_rewv1']
        policy_types = ['Final_Policy','BestPolicy']
        nu_intervals ={'1':[[0., 1], [0.10, 1], [0.90, 1.], [1., 1.]],
                       '2':[[0., 1], [0.80, 1], [0.90, 0.], [1., 0.]],
                       '3':[[0., 1], [0.70, 1], [0.80, 0.], [1., 0.]],
                       '4':[[0., 1], [0.60, 1], [0.70, 0.], [1., 0.]],
                       '5':[[0., 1], [0.50, 1], [0.60, 0.], [1., 0.]],
                       '6':[[0., 1], [0.40, 1], [0.50, 0.], [1., 0.]],
                       '7':[[0., 1], [0.30, 1], [0.40, 0.], [1., 0.]],
                       '8':[[0., 1], [0.20, 1], [0.30, 0.], [1., 0.]],
                       '9':[[0., 1], [0.10, 1], [0.20, 0.], [1., 0.]],
                       '10':[[0., 0], [0.10, 0], [0.20, 0.], [1., 0.]],
                       'original':[[0., 1], [0.30, 1], [0.60, 0.], [1., 0.]]}
        """for n in nu_intervals.keys():
            n='6'"""
        seeds = [17,43,45,3,31]
        #for seed in seeds:
        for nu_interval in nu_intervals.keys():
            if 'original' not in nu_interval:
                continue
            for policy_type in policy_types:
                if 'Final_Policy' not in policy_type:
                    continue
                for i,policy_name in enumerate(policy_names_veh4_first_paper):
                    if ('Experimento_serv_2_net_0'  not in policy_name) or ('one_agent_saves_in_buffer' in policy_name):
                    #if (('Experimento_serv_2_net_0_arch_v1_rew_v4'  not in policy_name) and ('Experimento_serv_2_net_0_arch_v2_rew_v2' not in policy_name)) or ('one_agent_saves_in_buffer' in policy_name):
                        continue
                    print(policy_name,nu_interval,policy_type)
                    #print(nu_intervals[nu_interval])
                    data_path1 = 'C:\\Users\\dames\\Downloads\\Learning\\runs\\Vehicles_4\\FirstPaper'
                    policy_path = f'{data_path1}/{policy_name}/'
                    seed = 30 #17#43#45#3#31#
                    """EvaluateMultiagent(number_of_agents=N,
                                    sc_map=sc_map,
                                    visitable_locations=visitable_locations,
                                    initial_positions=initial_positions,
                                    num_of_eval_episodes=num_of_eval_episodes,
                                    policy_path=policy_path,
                                    policy_type=policy_type+'.pth',
                                    seed=seed,
                                    policy_name=f'{policy_name}_{nu_interval}',
                                    metrics_directory= f'./Evaluation/Results/Results_seed30_firstpaper/{policy_type}',
                                    nu_interval = nu_intervals[nu_interval],
                                    render = False
                                    )
            """
            #plot_visits(imm,seed=seed)
            imm = []
    
    if False:
        reward_type = 'metrics global'
        ground_truth_type = 'algae_bloom'
        seed = 30

        algorithms = ['RandomWandering','LawnMower','GreedyAgent']
        for algorithm in algorithms:
            print(algorithm)
            env = MultiAgentPatrolling(scenario_map=sc_map,
                                fleet_initial_positions=initial_positions,
                                distance_budget=200,
                                number_of_vehicles=N,
                                seed=seed,
                                miopic=True,
                                detection_length=2,
                                movement_length=2,
                                max_collisions=np.inf,
                                forget_factor=0.5,
                                attrition=0.1,
                                reward_type='metrics global',
                                ground_truth_type=ground_truth_type,
                                obstacles=False,
                                frame_stacking=1,
                                state_index_stacking=(2, 3, 4),
                                reward_weights=(1.0, 0.1)
                                )
            run_path_planners_evaluation(path=f'./Evaluation/Results/Results_seed_{seed}_Heuristics/', 
                            env=env, 
                            algorithm = algorithm,
                            runs = num_of_eval_episodes,
                            n_agents = N, 
                            ground_truth_type = ground_truth_type, 
                            render = False,
                            save=True,
                            info={'seed': 0 })
    if False:            
        policy_types = ['Final_Policy','BestPolicy']#,'BestPolicy_reward_information','BestPolicy_reward_exploration']
        csvtype = ['DRL_paths.csv','DRLResults.csv']
        pathplaners_types = ['GreedyAgent','LawnMower','RandomWandering']
        data_path0 = 'Evaluation/Results/Results_seed_30_Heuristics'
        data_path1 = 'Evaluation/Results/Results_seed30_firstpaper'
        dff = []
        for pol in pathplaners_types:
            dff.append(pd.read_csv(f'{data_path}/{data_path0}/{pol}_Results.csv'))
        
        # Concatenate the two dataframes together
        result = pd.concat(dff)
        # Write the resulting dataframe to a new CSV file
        result.to_csv(f'{data_path}/{data_path1}/Heuristics_Results.csv', index=False)

    data_path0 = 'C:\\Users\\dames\\Downloads\\Learning\\runs\\Vehicles_4\\Finales\\Results_seed30_Finales'
    data_path1 = 'C:\\Users\\dames\\OneDrive\\Documentos\\GitHub\\MultiAgentPatrollingProblem\\Results_seed30_2rew'
    data_path2 = 'C:\\Users\\dames\\OneDrive\\Documentos\\GitHub\\MultiAgentPatrollingProblem\\Results_seed30_nu_intervals'
    data_path3 = 'C:\\Users\\dames\\OneDrive\\Documentos\\GitHub\\MultiAgentPatrollingProblem\\Results_seed30_firstpaper0'
    data_path4 = 'C:\\Users\\dames\\OneDrive\\Documentos\\GitHub\\MultiAgentPatrollingProblem\\Results_seed30_firstpaper0PER'
    data_path5 = 'C:\\Users\\dames\\OneDrive\\Documentos\\GitHub\\MultiAgentPatrollingProblem\\Evaluation\\Results\\Results_seed30_firstpaper'
    data_path6 = 'C:\\Users\\dames\\OneDrive\\Documentos\\GitHub\\MultiAgentPatrollingProblem\\Results_seed30_firstpaper_nu_intervals'
    data_path7 = 'C:\\Users\\dames\\OneDrive\\Documentos\\GitHub\\MultiAgentPatrollingProblem\\Evaluation\\Results_seed_30_Heuristics'
    data_path8 = 'C:\\Users\\dames\\OneDrive\\Documentos\\GitHub\\MultiAgentPatrollingProblem\\Evaluation\\Evaluation\\Results\\Results_seed_30_Heuristics'
    pol_resul = 'Final_PolicyDRLResults.csv'
    pol_resul = 'Heuristics_Results.csv'
    pol_resul = 'LawnMower_Results.csv'
    Finalpolicy = pd.read_csv(f'{data_path7}/{pol_resul}')
    #Finalpolicy0 = pd.read_csv(f'{data_path0}/{pol_resul}')
    indexes_to_not_skip = [ ]
    indexes_to_skip = [  
                        'Experimento_serv_27__net_0_arch_v1_rewv4_WLU',
                        'Experimento_serv_27__net_0_arch_v2_rewv4_WLU',
                        'Experimento_serv_27__net_4_arch_v1_rewv4_WLU', 
                        'Experimento_serv_27__net_4_arch_v2_rewv4_WLU',
                        'Experimento_serv_27__net_0_arch_v2_rewv2_v3_IGIdiv_m10col',
                        'Experimento_serv_27__net_0_arch_v1_rewv2_v3_IGI_div',
                        'Experimento_serv_27__net_0_arch_v2_rewv2_v3_IGI_div',
                        'Experimento_serv_27__net_4_arch_v1_rewv2_v3_IGI_div',
                        'Experimento_serv_27__net_4_arch_v2_rewv2_v3_IGI_div',]
           
    indexes_to_skip = [ ]          
    """'Experimento_serv_22__net_0_arch_v1_rewv2_no_cost',
                        'Experimento_serv_22__net_0_arch_v2_rewv2_no_cost',
                        'Experimento_serv_24_net_4_arch_v1_rew_v2',
                        'Experimento_serv_24_net_4_arch_v2_rew_v2',"""
    dictrename = {'Experimento_serv_22__net_0_arch_v1_rewv2_no_cost': 'Net0_arch0_RLocal',
                        'Experimento_serv_22__net_0_arch_v2_rewv2_no_cost': 'Net0_arch1_RLocal',
                        'Experimento_serv_24_net_4_arch_v1_rew_v2': 'Net1_arch0_RLocal',
                        'Experimento_serv_24_net_4_arch_v2_rew_v2': 'Net1_arch1_RLocal',
                        'Experimento_serv_27__net_0_arch_v1_rewv4_WLU': 'Net0_arch0_RDiff',
                        'Experimento_serv_27__net_0_arch_v2_rewv4_WLU':'Net0_arch1_RDiff',
                        'Experimento_serv_27__net_4_arch_v1_rewv4_WLU': 'Net1_arch0_RDiff',
                        'Experimento_serv_27__net_4_arch_v2_rewv4_WLU': 'Net1_arch1_RDiff'}
    dictrename = {'Experimento_serv_27__net_0_arch_v2_rewv4_WLU_1':'Only Exploration',
                  'Experimento_serv_27__net_0_arch_v2_rewv4_WLU_2':'80-90',
                  'Experimento_serv_27__net_0_arch_v2_rewv4_WLU_3':'70-80',
                  'Experimento_serv_27__net_0_arch_v2_rewv4_WLU_4':'60-70',
                  'Experimento_serv_27__net_0_arch_v2_rewv4_WLU_5':'50-60',
                  'Experimento_serv_27__net_0_arch_v2_rewv4_WLU_6':'40-50',
                  'Experimento_serv_27__net_0_arch_v2_rewv4_WLU_7':'30-40',
                  'Experimento_serv_27__net_0_arch_v2_rewv4_WLU_8':'20-30',
                  'Experimento_serv_27__net_0_arch_v2_rewv4_WLU_9':'10-20',
                  'Experimento_serv_27__net_0_arch_v2_rewv4_WLU_10':'Only Intensification',}
    dictrename = {'Experimento_serv_2_net_0_arch_v1_rew_v4':'Arch-v1 DR',
                  'Experimento_serv_2_net_0_arch_v2_rew_v4':'Arch-v2 DR',
                  'Experimento_serv_2_net_0_arch_v1_rew_v2':'Arch-v1 LR',
                  'Experimento_serv_2_net_0_arch_v2_rew_v2':'Arch-v2 LR'}
    
    dictrename = {'Experimento_serv_2_net_0_arch_v1_rew_v4_1':'Only Exploration',
                  'Experimento_serv_2_net_0_arch_v1_rew_v4_2':'80-90',
                  'Experimento_serv_2_net_0_arch_v1_rew_v4_3':'70-80',
                  'Experimento_serv_2_net_0_arch_v1_rew_v4_4':'60-70',
                  'Experimento_serv_2_net_0_arch_v1_rew_v4_5':'50-60',
                  'Experimento_serv_2_net_0_arch_v1_rew_v4_6':'40-50',
                  'Experimento_serv_2_net_0_arch_v1_rew_v4_7':'30-40',
                  'Experimento_serv_2_net_0_arch_v1_rew_v4_8':'20-30',
                  'Experimento_serv_2_net_0_arch_v1_rew_v4_9':'10-20',
                  'Experimento_serv_2_net_0_arch_v1_rew_v4_10':'Only Intensification',}
    dictrename = {'Experimento_serv_2_net_0_arch_v2_rew_v2_1':'Only Exploration',
                  'Experimento_serv_2_net_0_arch_v2_rew_v2_2':'80-90',
                  'Experimento_serv_2_net_0_arch_v2_rew_v2_3':'70-80',
                  'Experimento_serv_2_net_0_arch_v2_rew_v2_4':'60-70',
                  'Experimento_serv_2_net_0_arch_v2_rew_v2_5':'50-60',
                  'Experimento_serv_2_net_0_arch_v2_rew_v2_6':'40-50',
                  'Experimento_serv_2_net_0_arch_v2_rew_v2_7':'30-40',
                  'Experimento_serv_2_net_0_arch_v2_rew_v2_8':'20-30',
                  'Experimento_serv_2_net_0_arch_v2_rew_v2_9':'10-20',
                  'Experimento_serv_2_net_0_arch_v2_rew_v2_10':'Only Intensification'}
    
    indexes_to_skip = ['Experimento_serv_2_net_0_arch_v2_rew_v2_1',
                  'Experimento_serv_2_net_0_arch_v2_rew_v2_2',
                  'Experimento_serv_2_net_0_arch_v2_rew_v2_3',
                  'Experimento_serv_2_net_0_arch_v2_rew_v2_4',
                  'Experimento_serv_2_net_0_arch_v2_rew_v2_5',
                  'Experimento_serv_2_net_0_arch_v2_rew_v2_6',
                  'Experimento_serv_2_net_0_arch_v2_rew_v2_7',
                  'Experimento_serv_2_net_0_arch_v2_rew_v2_8',
                  'Experimento_serv_2_net_0_arch_v2_rew_v2_9',
                  'Experimento_serv_2_net_0_arch_v2_rew_v2_10']
    indexes_to_skip = ['Experimento_serv_2_net_0_arch_v1_rew_v4_1',
                  'Experimento_serv_2_net_0_arch_v1_rew_v4_2',
                  'Experimento_serv_2_net_0_arch_v1_rew_v4_3',
                  'Experimento_serv_2_net_0_arch_v1_rew_v4_4',
                  'Experimento_serv_2_net_0_arch_v1_rew_v4_5',
                  'Experimento_serv_2_net_0_arch_v1_rew_v4_6',
                  'Experimento_serv_2_net_0_arch_v1_rew_v4_7',
                  'Experimento_serv_2_net_0_arch_v1_rew_v4_8',
                  'Experimento_serv_2_net_0_arch_v1_rew_v4_9',
                  'Experimento_serv_2_net_0_arch_v1_rew_v4_10']
    dictrename = {'Experimento_serv_2_net_0_arch_v1_rew_v4_original':'Arch-v1 DR',
                'Experimento_serv_2_net_0_arch_v2_rew_v4_original':'Arch-v2 DR',
                'Experimento_serv_2_net_0_arch_v1_rew_v2_original':'Arch-v1 LR',
                'Experimento_serv_2_net_0_arch_v2_rew_v2_original':'Arch-v2 LR'}
    Finalpolicy = Finalpolicy[~Finalpolicy['Policy Name'].isin(indexes_to_skip)]
    #Finalpolicy['Policy Name'] = Finalpolicy['Policy Name'].map(dictrename)
    Finalpolicy.rename(columns={
                  'Accumulated Reward Intensification': 'Mean Weighted Idleness Intensification',
                  'Accumulated Reward Exploration': 'Mean Weighted Idleness Exploration',
                  'Average global idleness Intensification': 'Average Global Idleness Intensification',
                  'Average global idleness Exploration': 'Average Global Idleness Exploration',
                  'Total Accumulated Reward': 'Mean Weighted Idleness'}, inplace=True)
    #Finalpolicy0 = Finalpolicy0[Finalpolicy0['Policy Name'].isin(indexes_to_not_skip)]
    #Finalpolicy = pd.concat([Finalpolicy0, Finalpolicy], ignore_index=True)
    #Finalpolicy = Finalpolicy0
    values_to_evaluate =['Mean Weighted Idleness Intensification',
                        'Mean Weighted Idleness Exploration',
                        'Mean Weighted Idleness',
                        'Total Length',
                        'Total Collisions',
                        'Average Global Idleness Intensification',
                        'Average Global Idleness Exploration',
                        'Sum global idleness Intensification',
                        'Percentage Visited Exploration',
                        'Percentage Visited']
    Accum_per_episode = Finalpolicy.groupby(['Policy Name','Run'])[values_to_evaluate].tail(1) 
    #print(Accum_per_episode.to_markdown(),'\n \n \n')
    # merge the result dataframe with the original dataframe on the 'group' and 'value' columns
    Finalpolicy_accum = Finalpolicy.loc[Accum_per_episode.index]

    #print(Finalpolicy_accum.to_markdown(),'\n \n \n')
    Mean_per_episode = Finalpolicy_accum.groupby('Policy Name')[values_to_evaluate].median()
    # Filter out rows based on their index
    #Mean_per_episode = Mean_per_episode[~Mean_per_episode.index.isin(indexes_to_skip)]
    with open("output.txt", "a") as f:
        print(Mean_per_episode.sort_values('Mean Weighted Idleness Exploration',ascending=True).to_markdown(),'\n \n \n', file=f)
        # para pasar a latex
        #print(Mean_per_episode.sort_values('Mean Weighted Idleness Exploration',ascending=True).style.to_latex(),'\n \n \n', file=f)
    from pymoo.factory import get_performance_indicator
    from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

    # extract the two objective values from the dataframe into a numpy array
    objs = - Mean_per_episode[['Mean Weighted Idleness Intensification', 'Mean Weighted Idleness Exploration',"Percentage Visited Exploration","Percentage Visited"]].to_numpy()
    objs =  Mean_per_episode[["Average Global Idleness Intensification","Average Global Idleness Exploration"]].to_numpy()


    nds = NonDominatedSorting()
    fronts = nds.do(objs)

    # get the solutions in the first front (i.e., the Pareto front)
    pf = objs[fronts[0]]

    # print the Pareto front
    with open("output.txt", "a") as f:
        print("Pareto front:", file=f)
        print(pf,file=f)
    
    vals = ["Percentage Visited Exploration","Percentage Visited", "Accumulated Reward Intensification", "Accumulated Reward Exploration",
            "Average global idleness Intensification","Average global idleness Exploration"]
    vals = ["Percentage Visited Exploration","Percentage Visited",'Mean Weighted Idleness Intensification','Mean Weighted Idleness Exploration',
            'Mean Weighted Idleness', "Average Global Idleness Intensification","Average Global Idleness Exploration"]
    for val in vals:
        my_order =Mean_per_episode.sort_values(val,ascending=False).index
        plt.figure(figsize=(20,10))
        sns.set_style("whitegrid")
        sns.set(font_scale=1.8)
        ax=sns.boxplot(
        data=Finalpolicy_accum,
        x='Policy Name', y=val, hue='Policy Name',order=my_order,dodge=False
    )
        """if 'Mean Weighted Idleness Intensification' in val:
            current_ylim = ax.get_ylim()
            ax.set_ylim((current_ylim[0], current_ylim[1]*2))"""
        plt.title(val)
        plt.legend(fontsize = "25")
        plt.title(val, fontsize = "40")
        plt.ylabel(val, fontsize = "30")
        plt.show()
        #plt.savefig(f'{data_path5}/imagenes/{val}.png',bbox_inches='tight')
        plt.close()

# to print with colorbar 
"""fig,ax=plt.subplots()
im = ax.imshow(env.im1.get_array(),cmap='rainbow_r',vmin=0,vmax=1.0)
plt.colorbar(im,ax=ax)"""