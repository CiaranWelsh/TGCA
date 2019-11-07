import pickle
# add root and source to pythonpath
import site
import os
from operator import attrgetter
from collections import Counter
from deap.algorithms import varAnd
import sys

site.addsitedir(os.path.dirname(os.path.dirname(__file__)))

import random
from copy import deepcopy
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Sequence
from itertools import repeat
from sklearn.decomposition.pca import PCA
import umap

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from tgca import *


def get_data():
    data = pd.read_csv(PROTEOME_DATA_FILE)
    data = data.drop(['Cancer_Type', 'SetID'], axis=1)
    data = data.set_index(['Sample_ID', 'Sample_Type'], append=True)
    data = data[sorted(data.columns)]
    data = data.dropna(how='any', axis=1)
    return data


def evaluate(individual, data, n_clusters, n_init=30, n_jobs=6):
    counted = Counter(individual)
    size = len(individual)
    penalty_factor = 1 / size
    penalty = 0.0
    for num, count in counted.items():
        if count != 1:
            penalty += penalty_factor

    data = data.iloc[:, individual]
    kmeans = KMeans(n_clusters=n_clusters, n_init=n_init, n_jobs=n_jobs)
    kmeans.fit(data)
    score = silhouette_score(data, kmeans.labels_)
    return score - penalty,


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
    return [np.random.choice(individuals, replace=False) for i in range(k)]


def selTournament(individuals, k, tournsize, fit_attr="fitness"):
    """Select the best individual among *tournsize* randomly chosen
    individuals, *k* times. The list returned contains
    references to the input *individuals*.

    :param individuals: A list of individuals to select from.
    :param k: The number of individuals to select.
    :param tournsize: The number of individuals participating in each tournament.
    :param fit_attr: The attribute of individuals to use as selection criterion
    :returns: A list of selected individuals.

    """
    chosen = []
    for i in range(k):
        aspirants = [np.random.choice(range(len(individuals)), replace=False) for i in range(tournsize)]
        aspirants = [individuals[i] for i in aspirants]
        chosen.append(max(aspirants, key=attrgetter(fit_attr)))
    return chosen


def cxOnePoint(ind1, ind2):
    """Executes a one point crossover on the input :term:`sequence` individuals.
    The two individuals are modified in place. The resulting individuals will
    respectively have the length of the other.

    :param ind1: The first individual participating in the crossover.
    :param ind2: The second individual participating in the crossover.
    :returns: A tuple of two individuals.

    This function uses the :func:`~random.randint` function from the
    python base :mod:`random` module.
    """
    size = min(len(ind1), len(ind2))
    cxpoint = random.randint(1, size - 1)
    ind1[cxpoint:], ind2[cxpoint:] = ind2[cxpoint:], ind1[cxpoint:]

    return ind1, ind2


def mutUniformInt(individual, low, up, indpb):
    """Mutate an individual by replacing attributes, with probability *indpb*,
    by a integer uniformly drawn between *low* and *up* inclusively.

    :param individual: :term:`Sequence <sequence>` individual to be mutated.
    :param low: The lower bound or a :term:`python:sequence` of
                of lower bounds of the range from wich to draw the new
                integer.
    :param up: The upper bound or a :term:`python:sequence` of
               of upper bounds of the range from wich to draw the new
               integer.
    :param indpb: Independent probability for each attribute to be mutated.
    :returns: A tuple of one individual.
    """
    size = len(individual)
    if not isinstance(low, Sequence):
        low = repeat(low, size)
    elif len(low) < size:
        raise IndexError("low must be at least the size of individual: %d < %d" % (len(low), size))
    if not isinstance(up, Sequence):
        up = repeat(up, size)
    elif len(up) < size:
        raise IndexError("up must be at least the size of individual: %d < %d" % (len(up), size))

    for i, xl, xu in zip(range(size), low, up):
        if random.random() < indpb:
            individual[i] = random.randint(xl, xu)

    return individual,


def mutUniformFromOptions(individual, data, indpb):
    """Mutate an individual by replacing attributes, with probability *indpb*,
    by a integer uniformly drawn between *low* and *up* inclusively.

    :param individual: :term:`Sequence <sequence>` individual to be mutated.
    :param low: The lower bound or a :term:`python:sequence` of
                of lower bounds of the range from wich to draw the new
                integer.
    :param up: The upper bound or a :term:`python:sequence` of
               of upper bounds of the range from wich to draw the new
               integer.
    :param indpb: Independent probability for each attribute to be mutated.
    :returns: A tuple of one individual.
    """
    # print('mutating', individual, indpb, list(data.columns))
    size = len(individual)
    options = list(set(individual).difference(set(data.columns)))
    for i in range(size):
        if random.random() < indpb:
            individual[i] = np.random.choice(options)

    return individual,


def varAnd(population, toolbox, cxpb, mutpb):
    """Part of an evolutionary algorithm applying only the variation part
    (crossover **and** mutation). The modified individuals have their
    fitness invalidated. The individuals are cloned so returned population is
    independent of the input population.

    :param population: A list of individuals to vary.
    :param toolbox: A :class:`~deap.base.Toolbox` that contains the evolution
                    operators.
    :param cxpb: The probability of mating two individuals.
    :param mutpb: The probability of mutating an individual.
    :returns: A list of varied individuals that are independent of their
              parents.

    The variation goes as follow. First, the parental population
    :math:`P_\mathrm{p}` is duplicated using the :meth:`toolbox.clone` method
    and the result is put into the offspring population :math:`P_\mathrm{o}`.  A
    first loop over :math:`P_\mathrm{o}` is executed to mate pairs of
    consecutive individuals. According to the crossover probability *cxpb*, the
    individuals :math:`\mathbf{x}_i` and :math:`\mathbf{x}_{i+1}` are mated
    using the :meth:`toolbox.mate` method. The resulting children
    :math:`\mathbf{y}_i` and :math:`\mathbf{y}_{i+1}` replace their respective
    parents in :math:`P_\mathrm{o}`. A second loop over the resulting
    :math:`P_\mathrm{o}` is executed to mutate every individual with a
    probability *mutpb*. When an individual is mutated it replaces its not
    mutated version in :math:`P_\mathrm{o}`. The resulting :math:`P_\mathrm{o}`
    is returned.

    This variation is named *And* beceause of its propention to apply both
    crossover and mutation on the individuals. Note that both operators are
    not applied systematicaly, the resulting individuals can be generated from
    crossover only, mutation only, crossover and mutation, and reproduction
    according to the given probabilities. Both probabilities should be in
    :math:`[0, 1]`.
    """
    offspring = [toolbox.clone(ind) for ind in population]

    # to ensure randomness
    np.random.shuffle(offspring)

    # Apply crossover and mutation on the offspring
    for i in range(1, len(offspring), 2):
        if random.random() < cxpb:
            offspring[i - 1], offspring[i] = toolbox.mate(offspring[i - 1],
                                                          offspring[i])
            del offspring[i - 1].fitness.values, offspring[i].fitness.values
            # if len(offspring[i - 1]) != len(set(offspring[i - 1])):
            #     raise ValueError(
            #         'Offspring i-1 has duplicates resulting from mating: {}'.format(sorted(offspring[i - 1])))
            # if len(offspring[i]) != len(set(offspring[i])):
            #     raise ValueError('Offspring i has duplicates resulting from mating: {}'.format(sorted(offspring[1])))

    for i in range(len(offspring)):
        if random.random() < mutpb:
            offspring[i], = toolbox.mutate(offspring[i])
            del offspring[i].fitness.values
            # if len(offspring[i]) != len(set(offspring[i])):
            #     raise ValueError(
            #         'Offspring i-1 has duplicates resulting from mutating: {}'.format(sorted(offspring[i])))

    return offspring


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
    import time
    # Begin the generational process
    for gen in range(1, ngen + 1):
        gen_start_time = time.now()

        # Select the next generation individuals
        offspring = toolbox.select(population, len(population))

        # Vary the pool of individuals
        offspring = varAnd(offspring, toolbox, cxpb, mutpb)
        # individuals that were varied have their fitness invalidated

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
        duration = time.time()-gen_start_time
        avg_duration = avg_duration + 1.0 / i * (duration - avg_duration)

        logbook.record(gen=gen, nevals=len(invalid_ind), gen_duration=duration, avg_duration=avg_duration,
                       expected_end=(ngen+1-i)*avg_duration, **record)
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


def initRepeat(container, func, n):
    """Call the function *container* with a generator function corresponding
    to the calling *n* times the function *func*.

    :param container: The type to put in the data from func.
    :param func: The function that will be called n times to fill the
                 container.
    :param n: The number of times to repeat func.
    :returns: An instance of the container filled with data from func.

    This helper function can be used in conjunction with a Toolbox
    to register a generator of filled containers, as individuals or
    population.

        >>> import random
        >>> random.seed(42)
        >>> initRepeat(list, random.random, 2) # doctest: +ELLIPSIS,
        ...                                    # doctest: +NORMALIZE_WHITESPACE
        [0.6394..., 0.0250...]

    See the :ref:`list-of-floats` and :ref:`population` tutorials for more examples.
    """
    return container(func() for _ in range(n))


def read_pickle(f):
    if os.path.getsize(f) == 0:
        raise ValueError('Pickle file "{}" is empty'.format(f))
    with open(f, 'rb') as f2:
        results = pickle.load(f2)
    return results


class PlotterBase:
    """
    Shared base for plotter functions. Requires the same parameters used for
    k-means during features selection so the clustering can be reproduced.
    """

    def __init__(self, individual, data, n_clusters, n_init=30,
                 n_jobs=6, plot_pca=False, filename=None,
                 ncol=1):
        self.ncol = ncol
        self.filename = filename
        self.n_jobs = n_jobs
        self.plot_pca = plot_pca
        self.n_init = n_init
        self.n_clusters = n_clusters
        self.data = data
        self.individual = individual

        if isinstance(self.individual, list) and len(self.individual) == 1:
            self.individual = self.individual[0]
        self.data = self.data.iloc[:, self.individual]
        self.kmeans = KMeans(n_clusters=self.n_clusters,
                             n_init=self.n_init, n_jobs=self.n_jobs)
        self.kmeans.fit(self.data)
        # self.data.loc[:, 'class'] = self.kmeans.labels_
        class_df = pd.Series(self.kmeans.labels_, index=self.data.index, name='class')
        self.data = pd.concat([self.data, class_df], axis=1)
        self.data.set_index('class', append=True, inplace=True)
        self.score = silhouette_score(self.data, self.kmeans.labels_)

    def plot(self):
        raise NotImplementedError


class PCAPlotter(PlotterBase):

    def plot(self):
        pca = PCA(n_components=2)
        df = pca.fit_transform(self.data.values)
        df = pd.DataFrame(df, index=self.data.index)
        ids = sorted(list(self.data.columns))

        df = df.reset_index()
        fig = plt.figure()
        ax = sns.scatterplot(x=0, y=1, data=df, hue='class', edgecolor='black')
        sns.despine(fig=fig, top=True, right=True)
        plt.xlabel('PC1 ({}%)'.format(round(pca.explained_variance_ratio_[0], 4) * 100))
        plt.ylabel('PC2 ({}%)'.format(round(pca.explained_variance_ratio_[1], 4) * 100))
        from functools import reduce
        annot = reduce(lambda x, y: f'{x}\n{y}', ids)
        plt.title("Silhouette Score={}".format(round(self.score, 3)))
        plt.annotate('Proteins\n' + annot, xycoords=ax.transAxes, xy=(1, 0.1))
        if self.filename is None:
            plt.show()
        else:
            plt.savefig(self.filename, dpi=300, bbox_inches='tight')
            print('Results saved to "{}"'.format(self.filename))


class UMAPPlotter(PlotterBase):

    def plot(self, n_neighbors=5, min_dist=0.3, metric='correlation'):
        embedding = umap.UMAP(n_neighbors=n_neighbors,
                              min_dist=min_dist, metric=metric).fit_transform(
            self.data.values)
        print(embedding)
        # df = pd.DataFrame(df, index=data.index)
        #
        # df = df.reset_index()
        # fig = plt.figure()
        # sns.scatterplot(x=0, y=1, data=df, hue='class', edgecolor='black')
        # sns.despine(fig=fig, top=True, right=True)
        # plt.xlabel('PC1 ({}%)'.format(round(pca.explained_variance_ratio_[0], 4) * 100))
        # plt.ylabel('PC2 ({}%)'.format(round(pca.explained_variance_ratio_[1], 4) * 100))
        # plt.title("Silhouette Score={}".format(round(self.score, 3)))
        # if self.filename is None:
        #     plt.show()
        # else:
        #     plt.savefig(self.filename, dpi=300, bbox_inches='tight')


class DistributionPlotter(PlotterBase):

    def plot(self):
        """
        Plot array of distributions tiled on a canvas
        :return:
        """
        pass
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
    #####################################################
    #  command line configuration                       #
    #####################################################

    # begin setup of components needed for genetic algorithm
    # note these need to be created early so I brought them to the begining
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    # commands for the runner program
    runner_parser = subparsers.add_parser('runner',
                                          description='Run genetic algorithm feature selection for k means clustering of RPPA data')
    runner_parser.set_defaults(mode='runner', num_features=None, num_clusters=None)

    runner_parser.add_argument('num_clusters', type=int, help='Number of clusters to use in kmeans algorithm')
    runner_parser.add_argument('num_features', type=int, help='Number of features to use in clustering')
    runner_parser.add_argument('--num_jobs', nargs='?', default=6, type=int, help='Number of jobs to use in clustering')
    runner_parser.add_argument('--num_generations', nargs='?', default=50, type=int,
                               help='Number of generations to evolve')
    runner_parser.add_argument('--num_population', nargs='?', default=300, type=int,
                               help='Number of individuals in population')
    runner_parser.add_argument('--num_init', nargs='?', default=30, type=int,
                               help='Number of random initialisations to use in k means')
    runner_parser.add_argument('--cxpb', nargs='?', default=0.5, type=float,
                               help='The probability of mating two individuals')
    runner_parser.add_argument('--mutpb', nargs='?', default=0.2, type=float,
                               help='The probability of mutating an individual')
    runner_parser.add_argument('--tourny', nargs='?', default=0.2, type=float,
                               help='The proportion of individuals that are '
                                    'randomly selected for the tournment selection'
                                    'operator')

    runner_parser.add_argument('--run', nargs='?', default=True, type=bool, help='Whether to run the problem or not')
    plotter_parser = subparsers.add_parser('plotter', description='Plot the results')
    plotter_parser.set_defaults(mode='plotter')

    plotter_parser.add_argument('--num_clusters', type=int, help='Number of clusters to use in kmeans algorithm',
                                required='--pca' in sys.argv or '--umap' in sys.argv or '--tsne' in sys.argv)
    plotter_parser.add_argument('--num_features', type=int, help='Number of features to use in clustering',
                                required='--pca' in sys.argv or '--umap' in sys.argv or '--tsne' in sys.argv)
    plotter_parser.add_argument('--pca', help='Cluster with k means then do dimentionality reduction '
                                              'using PCA to visualise the clusters in 2D',
                                action='store_true')
    plotter_parser.add_argument('--umap', help='Cluster with k means then do dimentionality reduction '
                                               'using UMAP to visualise the clusters in 2D',
                                action='store_true')
    plotter_parser.add_argument('--tsne', help='Cluster with k means then do dimentionality reduction '
                                               'using TSNE to visualise the clusters in 2D',
                                action='store_true')
    plotter_parser.add_argument('--elbow', help='K on the x-axis with silhouette score on the y',
                                action='store_true')

    args = parser.parse_args()
    print(args)

    # the number of features to use for clustering
    args.num_features = args.num_features if 'num_features' in args else None
    # the number of clusters to use in kmeans
    args.num_clusters = args.num_clusters if 'num_clusters' in args else None

    if args.mode == 'runner':
        if args.tourny < 0 or args.tourny > 1:
            raise ValueError('tourny argument must be between 0 and 1. Got {}'.format(args.tourny))
        # need to make these available so they dont get run
        setattr(args, 'pca', False)
        setattr(args, 'tsne', False)
        setattr(args, 'umap', False)

    elif args.mode == 'plotter':
        print('Executing in plotter mode')
        # if were plotting a single plot, then use num_cluster and num_features to read the pickle
        pickle_file = os.path.join(GENETIC_ALGORITHM_PICKLES_DIR,
                                   f'features_{args.num_features}_clusters_{args.num_clusters}.pickle')
        if not os.path.isfile(pickle_file):
            raise FileNotFoundError(f'"{pickle_file}". Check your n_features and n_clusters arguments')
        # if we are doing a larger run comparison plot then need to do something else. Perhaps another argument parser
        dct = read_pickle(pickle_file)

        # set the attributes that were used to generate the pickle file
        setattr(args, 'num_init', dct['num_init'] if 'num_init' in dct else dct['N_INIT'])
        setattr(args, 'num_jobs', dct['num_jobs'] if 'num_jobs' in dct else dct['N_JOBS'])
        setattr(args, 'num_generations', dct['num_generations'] if 'num_generations' in dct else dct['N_GENERATIONS'])
        setattr(args, 'num_population', dct['num_population'] if 'num_population' in dct else dct['N_POPULATION'])
        setattr(args, 'cxpb', dct['cxpb'] if 'cxpb' in dct else dct['CXPB'])
        setattr(args, 'mutpb', dct['mutpb'] if 'mutpb' in dct else dct['MUTPB'])
        setattr(args, 'tourny', dct['tourny'] if 'tourny' in dct else dct['TOURNAMENT_SELECTION'])
        setattr(args, 'run', False if 'run' in dct else dct['RUN_GA'])

    # read our data in
    data = get_data()
    rows = data.index
    cols = data.columns

    # configure the DEAP toolbox
    toolbox = base.Toolbox()

    # Attribute generator
    toolbox.register("value", random.randint, 0, data.shape[1] - 1)

    # Structure initializers
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.value, args.num_features)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", evaluate, data=data, n_clusters=args.num_clusters, n_init=args.num_init,
                     n_jobs=args.num_jobs)
    toolbox.register("mate", cxOnePoint)
    toolbox.register("mutate", mutUniformFromOptions, data=data, indpb=0.05)
    selection_size = np.floor(args.num_population * args.tourny)
    toolbox.register("select", selTournament, tournsize=int(selection_size) if selection_size > 0 else 1)
    if args.mode == 'runner' and args.run:
        print('Executing in "runner" mode')
        pop, log, hof = ga(toolbox, population_size=args.num_population, number_generations=args.num_generations,
                           cxpb=args.cxpb,
                           mutpb=args.mutpb)
        result = dict(pop=pop, log=log, hof=hof,
                      num_features=args.num_features,
                      num_clusters=args.num_clusters,
                      num_jobs=args.num_jobs,
                      num_init=args.num_init,
                      num_generations=args.num_generations,
                      num_population=args.num_population,
                      cxpb=args.cxpb,
                      mutpb=args.mutpb,
                      tourny=args.tourny,
                      run=args.run,
                      )

        fname = os.path.join(GENETIC_ALGORITHM_PICKLES_DIR,
                             f'features_{args.num_features}_clusters_{args.num_clusters}.pickle')

        with open(fname, 'wb') as f:
            pickle.dump(result, f)

    if args.pca or args.tsne or args.umap:
        print(dct.keys())
        best_individual = creator.Individual(dct['hof'])

    if args.pca:
        print('plotting with pca dim reduction')
        fname = os.path.join(GENETIC_ALGORITHM_PCA_PLOTS_DIR,
                             f'features_{args.num_features}_clusters_{args.num_clusters}.png')
        p = PCAPlotter(best_individual, data, n_clusters=args.num_clusters,
                       n_jobs=args.num_jobs, n_init=args.num_init, plot_pca=True,
                       filename=fname)
        p.plot()
    if args.umap:
        fname = os.path.join(GENETIC_ALGORITHM_UMAP_PLOTS_DIR,
                             f'features_{args.num_features}_clusters_{args.num_clusters}.png')
        p = UMAPPlotter(best_individual, data, n_clusters=args.num_clusters,
                        n_jobs=args.num_jobs, n_init=args.num_init, plot_pca=True,
                        filename=fname,
                        n_neighbors=5,
                        min_dist=0.3,
                        metric='correlation'
                        )
        p.plot()
