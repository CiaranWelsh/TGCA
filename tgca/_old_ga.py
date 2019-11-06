# add root and source to pythonpath
import site
import os
# site.addsitedir(os.path.dirname(__file__))
site.addsitedir(os.path.dirname(os.path.dirname(__file__)))

import array
import random

import numpy
import pandas as pd

from tgca import *

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def get_data():
    data = pd.concat([pd.read_csv(i) for i in PROTEOME_DATA_FILE], axis=0, sort=False)
    data = data.drop(['Cancer_Type', 'SetID'], axis=1)
    data = data.set_index(['Sample_ID', 'Sample_Type'], append=True)
    data = data[sorted(data.columns)]
    data = data.dropna(how='any', axis=1)
    return data


def evalOneMax(individual):
    return sum(individual),


def evaluate(individual, data, n_clusters, n_init=30, n_jobs=6):
    data = data[individual]
    kmeans = KMeans(n_clusters=n_clusters, n_init=n_init, n_jobs=n_jobs)
    kmeans.fit(data)
    score = silhouette_score(data, kmeans.labels_),
    # print(individual, score)
    return score


def ga(toolbox, population_size=300, number_generations=40,
       cxpb=0.5, mutpb=0.2):
    pop = toolbox.population(n=population_size)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)

    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=cxpb, mutpb=mutpb, ngen=number_generations,
                                   stats=stats, halloffame=hof, verbose=True)

    return pop, log, hof



if __name__ == "__main__":
    # the number of features to use for clustering
    NUM_FEATURES = 25
    # the number of clusters to use in kmeans
    NUM_CLUSTERS = 4
    # the number of jobs to use in kmeans
    N_JOBS = 6
    # the number of random initialisation to use in kmeans
    N_INIT = 30
    # the number of generations
    N_GENERATIONS = 40
    # population size
    N_POPULATION = 200

    data = get_data()
    rows = data.index
    cols = data.columns
    data.columns = range(data.shape[1])

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
    toolbox.register("select", tools.selTournament, tournsize=N_POPULATION // 10)

    # pop, log, hof = ga(toolbox, population_size=N_POPULATION, number_generations=N_GENERATIONS,
    #                    cxpb=0.5, mutpb=0.2)
    # print(pop)
    # print(log)
    # print(hof)
    # x = [77, 77, 77, 77, 77, 44, 45, 77, 204, 77]
    # print(data[x])
    # print(cols[x])

