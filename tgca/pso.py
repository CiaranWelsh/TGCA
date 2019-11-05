import operator
import random

import numpy as np
import pandas as pd
import math
from tgca import *

from deap import base
from deap import benchmarks
from deap import creator
from deap import tools
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from copy import deepcopy
"""
notes on PSO
------------
- Consist of a number of simple entities, called particles
- Each particle evaluates the objective function at its current location
    - Therefore each particle should have as many numbers as I want features in the clustering problem
    - This is not ideal, I Would like the number of features itself to be a parameter. 
- Each particle in the swarm has three D-dimensional vectors
    - The current position xi, the best position already seen pi and the velocity 

"""


def get_data():
    data = pd.concat([pd.read_csv(i) for i in PROTEOME_FILES_LEVEL4], axis=0, sort=False)
    data = data.drop(['Cancer_Type', 'SetID'], axis=1)
    data = data.set_index(['Sample_ID', 'Sample_Type'], append=True)
    data = data[sorted(data.columns)]
    data = data.dropna(how='any', axis=1)
    return data


creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Particle", list, fitness=creator.FitnessMax, speed=list,
               smin=None, smax=None, best=None)


def generate(size, pmin, pmax, smin, smax):
    part = creator.Particle(np.random.randint(pmin, pmax) for _ in range(size))
    part.speed = [np.random.randint(smin, smax) for _ in range(size)]
    part.smin = smin
    part.smax = smax
    return part


def updateParticle(part, best, phi1, phi2, size):
    initial_state = deepcopy(part)
    u1 = (np.random.randint(-phi1, phi1) for _ in range(len(part)))
    u2 = (np.random.randint(-phi2, phi2) for _ in range(len(part)))
    v_u1 = map(operator.mul, u1, map(operator.sub, part.best, part))
    v_u2 = map(operator.mul, u2, map(operator.sub, best, part))
    part.speed = list(map(operator.add, part.speed, map(operator.add, v_u1, v_u2)))
    # print('x', part, part.speed)
    for i, speed in enumerate(part.speed):
        if abs(speed) < part.smin:
            part.speed[i] = math.copysign(part.smin, speed)
        elif abs(speed) > part.smax:
            part.speed[i] = math.copysign(part.smax, speed)
    part_candidate = list(map(operator.add, part, part.speed))
    # for i in part_candidate:
    #     if int(i) not in range(size):
    #         # if we can't use these numbers,
    #         # reinitialise the state
    #         part = initial_state
    #         updateParticle(part, best, phi1, phi2, size)
    part[:] = part_candidate


def example_function(individual, ):
    num = (np.sin(individual[0] - individual[1] / 8)) ** 2 + (np.sin(individual[1] + individual[0] / 8)) ** 2
    denum = ((individual[0] - 8.6998) ** 2 + (individual[1] - 6.7665) ** 2) ** 0.5 + 1
    return num / denum,


def evaluate(individual, data):
    try:
        data = data[individual]
    except KeyError:
        return -1,#1 / sum([abs(i) for i in individual]),
    kmeans = KMeans(n_clusters=4, n_init=30, n_jobs=6)
    kmeans.fit(data)
    return silhouette_score(data, kmeans.labels_),


def pso(toolbox, population_size=15, num_generations=1000):
    pop = toolbox.population(n=population_size)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    logbook = tools.Logbook()
    logbook.header = ["gen", "evals"] + stats.fields

    GEN = num_generations
    best = None
    for g in range(GEN):
        for part in pop:
            part.fitness.values = toolbox.evaluate(part)
            if not part.best or part.best.fitness < part.fitness:
                part.best = creator.Particle(part)
                part.best.fitness.values = part.fitness.values
            if not best or best.fitness < part.fitness:
                best = creator.Particle(part)
                best.fitness.values = part.fitness.values
        for part in pop:
            toolbox.update(part, best)

        # Gather all the fitnesses in one list and print the stats
        logbook.record(gen=g, evals=len(pop), **stats.compile(pop))
        print(logbook.stream)

    return pop, logbook, best


if __name__ == "__main__":
    data = get_data()
    colnames = list(data.columns)
    rownames = data.index

    data = data.reset_index(drop=True)
    data.columns = range(data.shape[1])

    toolbox = base.Toolbox()
    toolbox.register("particle", generate, size=20, pmin=0, pmax=data.shape[1]-1, smin=0, smax=data.shape[1]-1)
    toolbox.register("population", tools.initRepeat, list, toolbox.particle)
    toolbox.register("update", updateParticle, phi1=2.0, phi2=2.0, size=data.shape[1])
    toolbox.register("evaluate", evaluate, data=data)

    results = pso(toolbox)
    pop, log, best = results
    print(log)