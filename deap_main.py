
import operator
import math
import random
import numpy
from copy import deepcopy
import re

import torch
from torch import nn, tensor, Tensor
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp

from deap_primitives import get_primitive_set
from CIFAR10 import CIFAR10Data

from deap_utils import get_pre_trained_model
from deap_evolution import DeapEvolution

from neural_network import ConvNet_CIFAR10


device = 'cuda' if torch.cuda.is_available() else 'cpu'

# model = get_pre_trained_model(device)






train_dataloader = CIFAR10Data().train_dataloader()
test_dataloader = CIFAR10Data().test_dataloader()

inputs, classes = next(iter(train_dataloader))  
model = ConvNet_CIFAR10(inputs[0].shape)

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

pset = get_primitive_set()

de = DeapEvolution(model, pset, train_dataloader, test_dataloader)


toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=4)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

toolbox.register("evaluate", de.evaluate_AF_single_restart)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register('mutate', gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

def main():
    random.seed(318)

    pop = toolbox.population(n=10)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)

    algorithms.eaSimple(pop, toolbox, cxpb = 0.5, mutpb = 0.1, ngen = 3, stats = stats, halloffame=hof)

    return pop, stats, hof


pop, stats, hof = main()
print(str(hof[0]))









