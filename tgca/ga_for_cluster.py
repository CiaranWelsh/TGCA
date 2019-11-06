import pickle
# add root and source to pythonpath
import site
import os
from  operator import attrgetter

# site.addsitedir(os.path.dirname(__file__))
from deap.algorithms import varAnd

site.addsitedir(os.path.dirname(os.path.dirname(__file__)))

import random
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import seaborn as sns

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from tgca import *


def get_data():
    data = pd.concat([pd.read_csv(i) for i in PROTEOME_FILES_LEVEL4], axis=0, sort=False)
    data = data.drop(['Cancer_Type', 'SetID'], axis=1)
    data = data.set_index(['Sample_ID', 'Sample_Type'], append=True)
    data = data[sorted(data.columns)]
    data = data.dropna(how='any', axis=1)
    return data


def evaluate(individual, data, n_clusters, n_init=30, n_jobs=6):
    data = data.iloc[:, individual]
    kmeans = KMeans(n_clusters=n_clusters, n_init=n_init, n_jobs=n_jobs)
    kmeans.fit(data)
    score = silhouette_score(data, kmeans.labels_),
    return score

def selRandom(individuals, k):
    """Select *k* individuals at random from the input *individuals* with
    replacement. The list returned contains references to the input
    *individuals*.

    :param individuals: A list of individuals to select from.
    :param k: The number of individuals to select.
    :returns: A list of selected individuals.

    This function uses the :func:`~random.choice` function from the
    python base :mod:`random` module.
    """
    return [random.choice(individuals) for i in range(k)]

def selTournament(individuals, k, tournsize, fit_attr="fitness"):
    """Select the best individual among *tournsize* randomly chosen
    individuals, *k* times. The list returned contains
    references to the input *individuals*.

    :param individuals: A list of individuals to select from.
    :param k: The number of individuals to select.
    :param tournsize: The number of individuals participating in each tournament.
    :param fit_attr: The attribute of individuals to use as selection criterion
    :returns: A list of selected individuals.

    This function uses the :func:`~random.choice` function from the python base
    :mod:`random` module.
    """
    chosen = []
    for i in range(k):
        aspirants = selRandom(individuals, tournsize)
        chosen.append(max(aspirants, key=attrgetter(fit_attr)))
    return chosen


def eaSimple(population, toolbox, cxpb, mutpb, ngen, stats=None,
             halloffame=None, verbose=__debug__):
    """This algorithm reproduce the simplest evolutionary algorithm as
    presented in chapter 7 of [Back2000]_.

    :param population: A list of individuals.
    :param toolbox: A :class:`~deap.base.Toolbox` that contains the evolution
                    operators.
    :param cxpb: The probability of mating two individuals.
    :param mutpb: The probability of mutating an individual.
    :param ngen: The number of generation.
    :param stats: A :class:`~deap.tools.Statistics` object that is updated
                  inplace, optional.
    :param halloffame: A :class:`~deap.tools.HallOfFame` object that will
                       contain the best individuals, optional.
    :param verbose: Whether or not to log the statistics.
    :returns: The final population
    :returns: A class:`~deap.tools.Logbook` with the statistics of the
              evolution

    The algorithm takes in a population and evolves it in place using the
    :meth:`varAnd` method. It returns the optimized population and a
    :class:`~deap.tools.Logbook` with the statistics of the evolution. The
    logbook will contain the generation number, the number of evalutions for
    each generation and the statistics if a :class:`~deap.tools.Statistics` is
    given as argument. The *cxpb* and *mutpb* arguments are passed to the
    :func:`varAnd` function. The pseudocode goes as follow ::

        evaluate(population)
        for g in range(ngen):
            population = select(population, len(population))
            offspring = varAnd(population, toolbox, cxpb, mutpb)
            evaluate(offspring)
            population = offspring

    As stated in the pseudocode above, the algorithm goes as follow. First, it
    evaluates the individuals with an invalid fitness. Second, it enters the
    generational loop where the selection procedure is applied to entirely
    replace the parental population. The 1:1 replacement ratio of this
    algorithm **requires** the selection procedure to be stochastic and to
    select multiple times the same individual, for example,
    :func:`~deap.tools.selTournament` and :func:`~deap.tools.selRoulette`.
    Third, it applies the :func:`varAnd` function to produce the next
    generation population. Fourth, it evaluates the new individuals and
    compute the statistics on this population. Finally, when *ngen*
    generations are done, the algorithm returns a tuple with the final
    population and a :class:`~deap.tools.Logbook` of the evolution.

    .. note::

        Using a non-stochastic selection method will result in no selection as
        the operator selects *n* individuals from a pool of *n*.

    This function expects the :meth:`toolbox.mate`, :meth:`toolbox.mutate`,
    :meth:`toolbox.select` and :meth:`toolbox.evaluate` aliases to be
    registered in the toolbox.

    .. [Back2000] Back, Fogel and Michalewicz, "Evolutionary Computation 1 :
       Basic Algorithms and Operators", 2000.
    """
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    # Begin the generational process
    for gen in range(1, ngen + 1):
        # Select the next generation individuals
        offspring = toolbox.select(population, len(population))

        # Vary the pool of individuals
        offspring = varAnd(offspring, toolbox, cxpb, mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Replace the current population by the offspring
        population[:] = offspring

        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

    return population, logbook


def ga(toolbox, population_size=300, number_generations=40,
       cxpb=0.5, mutpb=0.2):
    pop = toolbox.population(n=population_size)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    pop, log = eaSimple(pop, toolbox, cxpb=cxpb, mutpb=mutpb, ngen=number_generations,
                        stats=stats, halloffame=hof, verbose=True)

    return pop, log, hof


def read_pickle(f):
    if os.path.getsize(f) == 0:
        raise ValueError('Pickle file "{}" is empty'.format(f))
    with open(f, 'rb') as f2:
        results = pickle.load(f2)
    return results


def evaluate_and_plot(individual, data, n_clusters, n_init=30,
                      n_jobs=6, plot_pca=False, filename=None):
    if isinstance(individual, list) and len(individual) == 1:
        individual = individual[0]
    data = data.iloc[:, individual]
    kmeans = KMeans(n_clusters=n_clusters, n_init=n_init, n_jobs=n_jobs)
    kmeans.fit(data)
    data.loc[:, 'class'] = kmeans.labels_
    data['class'] = data['class'].astype('int')
    data.set_index('class', append=True, inplace=True)
    score = silhouette_score(data, kmeans.labels_)

    # now do a pca so we can plot in 2d space
    from sklearn.decomposition.pca import PCA
    pca = PCA(n_components=2)
    df = pca.fit_transform(data.values)
    df = pd.DataFrame(df, index=data.index)

    if plot_pca:
        df = df.reset_index()
        fig = plt.figure()
        sns.scatterplot(x=0, y=1, data=df, hue='class', edgecolor='black')
        sns.despine(fig=fig, top=True, right=True)
        plt.xlabel('PC1 ({}%)'.format(round(pca.explained_variance_ratio_[0], 4) * 100))
        plt.ylabel('PC1 ({}%)'.format(round(pca.explained_variance_ratio_[1], 4) * 100))
        plt.title("Silhouette Score={}".format(round(score, 3)))
        if filename is None:
            plt.show()
        else:
            plt.savefig(filename, dpi=300, bbox_inches='tight')


    # work out how many rows to suit the number of columns
    # total = data.shape[1]
    # nrows = int(np.floor(total / ncols))
    # remainder = total % ncols
    # if remainder > 0:
    #     nrows += 1
    # print(score)
    # data = data.reset_index()
    # print(data.head())
    # fig = plt.figure()
    # for i, x in enumerate(plot_attrs):
    #     print(i, x)
    #     print(plot_attrs)
    #     ax = plt.subplot(nrows, ncols, i+1)
    #     # plot_data = data[x]
    #     sns.scatterplot(x='level_0', y=x, data=data)
    # plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('num_clusters', type=int, help='Number of clusters to use in kmeans algorithm')
    parser.add_argument('num_features', type=int, help='Number of features to use in clustering')
    parser.add_argument('--num_jobs', nargs='?', default=6, type=int, help='Number of jobs to use in clustering')
    parser.add_argument('--num_generations', nargs='?', default=50, type=int, help='Number of generations to evolve')
    parser.add_argument('--num_population', nargs='?', default=300, type=int,
                        help='Number of individuals in population')
    parser.add_argument('--num_init', nargs='?', default=30, type=int,
                        help='Number of random initialisations to use in k means')
    parser.add_argument('--cxpb', nargs='?', default=0.5, type=float, help='The probability of mating two individuals')
    parser.add_argument('--mutpb', nargs='?', default=0.2, type=float,
                        help='The probability of mutating an individuals')
    parser.add_argument('--tourny', nargs='?', default=0.1, type=float, help='The proportion of individuals that are '
                                                                             'randomly selected for the tournment selection'
                                                                             'operator')
    args = parser.parse_args()
    print(args)

    # the number of features to use for clustering
    NUM_FEATURES = args.num_features
    # the number of clusters to use in kmeans
    NUM_CLUSTERS = args.num_clusters
    # the number of jobs to use in kmeans
    N_JOBS = args.num_jobs
    # the number of random initialisation to use in kmeans
    N_INIT = args.num_init
    # the number of generations
    N_GENERATIONS = args.num_generations
    # population size
    N_POPULATION = args.num_population
    # propbability of two individuals mating
    CXPB = args.cxpb
    # probability of two individuals mutating
    MUTPB = args.mutpb
    # tournment proportion
    TOURNAMENT_SELECTION = args.tourny
    if TOURNAMENT_SELECTION < 0 or TOURNAMENT_SELECTION > 1:
        raise ValueError('tourny argument must be between 0 and 1. Got {}'.format(TOURNAMENT_SELECTION))

    # run the genetic algorithm
    RUN_GA = False
    # visualise some results
    VIZ_GA = True
    # pca PLOT
    VIZ_BY_PCA = False
    # el
    ELBOW = True


    data = get_data()
    rows = data.index
    cols = data.columns

    # begin setup
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    # Attribute generator
    toolbox.register("value", random.randint, 0, data.shape[1] - 1)

    # Structure initializers
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.value, NUM_FEATURES)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", evaluate, data=data, n_clusters=NUM_CLUSTERS, n_init=N_INIT, n_jobs=N_JOBS)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    selection_size = np.floor(N_POPULATION * TOURNAMENT_SELECTION)
    toolbox.register("select", selTournament, tournsize=int(selection_size) if selection_size > 0 else 1)


    if RUN_GA:
        pop, log, hof = ga(toolbox, population_size=N_POPULATION, number_generations=N_GENERATIONS, cxpb=CXPB,
                           mutpb=MUTPB)
        result = dict(pop=pop, log=log, hof=hof)

        fname = os.path.join(GENETIC_ALGORITHM_DATA_DIR, f'features_{NUM_FEATURES}_clusters_{NUM_CLUSTERS}.pickle')

        with open(fname, 'wb') as f:
            pickle.dump(result, f)

    if VIZ_GA:
        pickle_files = GENETIC_ALGORITHM_RESULTS_PICKLES

        res = read_pickle(pickle_files[0])
        print(res.keys())
        print(res['hof'], list(data.iloc[:, res['hof'][0]].columns))
        best_individual = creator.Individual(res['hof'])

        if VIZ_BY_PCA:
            evaluate_and_plot(best_individual, data, n_clusters=NUM_CLUSTERS,
                          n_jobs=N_JOBS, n_init=N_INIT, plot_pca=True)

        if ELBOW:
            print(pickle_files)
