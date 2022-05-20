
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
from deap_utils import create_root_tree

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

toolbox.register("evaluate", de.evaluate_AF_triple_restart)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register('mutate', gp.mutUniform, expr=toolbox.expr_mut, pset=pset)



from deap_utils import save_activation_location, replace_model_activations, activation_module
def condition(height, depth):
    """Expression generation stops when the depth is equal to height."""
    return depth == height

temp_model = deepcopy(model)
layer_dict = {}
save_activation_location(temp_model, layer_dict)

mapper = {0:range(0,1),1:range(1,2),2:range(2,3)}

POP_SIZE = 10
NUM_SPECIES = 3
NGEN = 3
CXPB = 0.2
MUTPB = 0.2
online_learning = False
CREATE_ORIGINAL_AF_IND=False
pop_coop = []
best_coop = []

for s in range(NUM_SPECIES):
    pop_coop.append(toolbox.population(n=POP_SIZE))
    for ind in pop_coop[s]:
        ind.fitness.values = toolbox.evaluate(ind, mapper[s])
    best_coop.append(tools.selBest(pop_coop[s], 1)[0])
    if online_learning:
        pass # set model at pop_coop[s] now

for g in range(1, NGEN):
    for s in range(NUM_SPECIES):
        # Select and clone the offspring
        offsprings_s = toolbox.select(pop_coop[s], len(pop_coop))
        offsprings_s = [toolbox.clone(ind) for ind in offsprings_s]        

        offsprings_s = algorithms.varAnd(offsprings_s, toolbox, cxpb = CXPB, mutpb = MUTPB)

        for ind in offsprings_s:
            ind.fitness.values = toolbox.evaluate(ind, mapper[s], best_coop)

        # Replace the old population by the offspring
        pop_coop[s] = offsprings_s
        best_coop[s] = tools.selBest(pop_coop[s], 1)[0]









