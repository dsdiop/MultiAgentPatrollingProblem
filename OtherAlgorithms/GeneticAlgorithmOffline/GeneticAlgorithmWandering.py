import numpy as np
from deap import base
from deap import creator
from deap import tools
import random
import datetime
import multiprocessing
import matplotlib.pyplot as plt
from StochasticEASimple import eaSimpleWithReevaluation, cxTwoPointCopy
from Environment.PatrollingEnvironments import MultiAgentPatrolling
from Evaluation.Utils.metrics_wrapper import MetricsDataCreator


""" Compute Ground Truth """
N = 4
sc_map = np.genfromtxt('../../Environment/example_map.csv', delimiter=',')
initial_positions = np.asarray([[24, 21], [28, 24], [27, 19], [24, 24]])
NUM_OF_GENERATIONS = 100
NUM_OF_INDIVIDUALS = 2000
NUM_OF_TRIALS = 10

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
                             algorithm_name='Genetic Algorithm',
                             experiment_name='GeneticAlgorithmNetworked',
                             directory='./')

paths = MetricsDataCreator(metrics_names=['vehicle', 'x', 'y'],
                           algorithm_name='Genetic Algorithm',
                           experiment_name='GeneticAlgorithmNetworked_paths',
                           directory='./')



# --- Create the Genetic Algorihtm Strategies --- #
creator.create("FitnessMax", base.Fitness, weights=(1.0,)) # This means we want to maximize
creator.create("Individual", np.ndarray, fitness=creator.FitnessMax) # Individual creator
# Create a toolbox for genetic operations #
toolbox = base.Toolbox()
# Cromosome is an action in [0,number_of_actions] #
toolbox.register("attr_bool", random.randint, 0, 7)
# Each individual is a set of n_agents x 101 steps (this will depend on the number of possible actions for agent)  #
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=N*201)
# Create the population creator #
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


def evalEnv(ind, local_env):
    """ Evaluate the environment. If the environment is stochastic, every evaluation will
    return a different value of reward. The best individual is that one that survives on average
    across a lot of different generations, which is, the strongest-average-one. """

    # Reset conditions #
    fitness = []
    for run in range(NUM_OF_TRIALS):

        R = 0
        local_env.reset()
        done = False

        # Slice the individual array into agents actions #
        # t0 -> [1,2,1,1]
        # t1 -> [0,1,3,6]
        # ...
        # tN -> [2,7,1,7]

        # Transform individual in agents actions #
        action_array = np.asarray(np.split(ind, N)).T
        action_indx = 0

        # If the initial action is valid, begin to evaluate #
        while not done:

            # ACT!
            _, r, done, _ = local_env.step(action_array[action_indx])
            action_indx += 1
            # Accumulate the reward into the fitness #
            R += np.sum(r)

        fitness.append(R)

    return np.mean(fitness),


toolbox.register("evaluate", evalEnv, local_env=env)
toolbox.register("mate", cxTwoPointCopy)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=5)

pool = multiprocessing.Pool()
toolbox.register("map", pool.map)


def optimize(local_env, save=False):


    random.seed(64)

    pop = toolbox.population(n=NUM_OF_INDIVIDUALS)

    # Fix the evaluation function with the current environment
    toolbox.register("evaluate", evalEnv, local_env=local_env)

    # Hall of fame for the TOP 5 individuals #
    hof = tools.HallOfFame(5, similar=np.array_equal)

    # Statistics #
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    # Algorithm - Simple Evolutionary Algorithm #
    eaSimpleWithReevaluation(pop, toolbox, cxpb=0.7, mutpb=0.3, ngen=NUM_OF_GENERATIONS, stats=stats, halloffame=hof)

    if save:

        with open(f"ga_simple_optimization_result_{datetime.datetime.now().strftime('%Y_%m_%d-%H:%M_%S')}.txt", "w") as solution_file:

            solution_file.write("Optimization result for the GA\n")
            solution_file.write("---------------------------------\n")
            solution_file.write("---------------------------------\n")
            solution_file.write("--------- Best Individuals -----------\n")

            for idx, individual in enumerate(hof):
                str_data = ','.join(str(i) for i in individual)
                solution_file.write(f"Individual {idx}: {str_data}\n")
                solution_file.write(f"Fitness: {individual.fitness.values}\n")

            solution_file.close()

    return hof[0]


if __name__ == "__main__":

    np.random.seed(0)

    # Create the environment to optimize #
    my_env = MultiAgentPatrolling(scenario_map=sc_map,
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

    # Initial reset #
    my_env.reset()

    """ OPTIMIZE THE SCENARIO """
    best = np.asarray(optimize(my_env, save=True))
    action_array = np.asarray(np.split(best, N)).T

    # Evaluate for 10 scenarios #
    for run in range(10):

        t = 0
        R = 0
        # Initial reset #
        my_env.reset()

        # Evaluate the metrics of the solutions #
        action_indx = 0
        done = False

        # If the initial action is valid, begin to evaluate #
        while not done:
            # ACT!
            t += 1
            _, r, done, _ = my_env.step(action_array[action_indx])
            action_indx += 1

            R = np.mean(r) + R

            # Register positions and metrics #
            metrics.register_step(run_num=run, step=t, metrics=[R, env.fleet.number_of_disconnections])
            for veh_id, veh in enumerate(env.fleet.vehicles):
                paths.register_step(run_num=run, step=t, metrics=[veh_id, veh.position[0], veh.position[1]])



            my_env.render()
            plt.pause(0.1)

metrics.register_experiment()
paths.register_experiment()


pool.close()
