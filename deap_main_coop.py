
import operator
import math
import random
import numpy as np
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
from deap_evolution import DeapEvolution
from neural_network import ConvNet_CIFAR10
from deap_net import get_test_score
from deap_utils import get_pre_trained_model
from deap_utils import create_root_tree, each_mapper
from deap_utils import save_activation_location, replace_model_activations, activation_module

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

toolbox.register("evaluate", de.evaluate_AF_coop)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register('mutate', gp.mutUniform, expr=toolbox.expr_mut, pset=pset)






layer_dict = {}
save_activation_location(model, layer_dict)

mapper = each_mapper(layer_dict)
N_RUNS = 1
POP_SIZE = 10
NUM_SPECIES = 3
NGEN = 3
CXPB = 0.4
MUTPB = 0.4
online_learning_AF = False
CREATE_ORIGINAL_AF_IND=True
online_learning_AF_WEIGHTS = True

name = 'CIFAR10_coop_3_layers_from_0'

try:
    res_dict = np.load('results/'+name+'.npy', allow_pickle=True).item()
except:
    res_dict = {}
    np.save('results/'+name+'.npy',res_dict, allow_pickle=True)

for n in range(N_RUNS):
    pop_coop = []
    best_coop = []
    model = ConvNet_CIFAR10(inputs[0].shape)
    de = DeapEvolution(model, pset, train_dataloader, test_dataloader)
    layer_dict = {}
    save_activation_location(model, layer_dict)

    for s in range(NUM_SPECIES):
        pop_coop.append(toolbox.population(n=POP_SIZE))
        if CREATE_ORIGINAL_AF_IND:
            for layer in mapper[s]:
                pop_coop[s].append(create_root_tree(layer_dict[layer][-1])[0])
        for ind in pop_coop[s]:
            ind.fitness.values, _ = toolbox.evaluate(ind, mapper[s])
        best_coop.append(tools.selBest(pop_coop[s], 1)[0])

    best_model = model
    best_model_loss = float('inf')
    res_dict[(n,'loss')] = []

    for g in range(1, NGEN):
        for s in range(NUM_SPECIES):
            # Select and clone the offspring
            offsprings_s = toolbox.select(pop_coop[s], len(pop_coop[s]))
            offsprings_s = [toolbox.clone(ind) for ind in offsprings_s]        
            offsprings_s = algorithms.varAnd(offsprings_s, toolbox, cxpb = CXPB, mutpb = MUTPB)
            models = []
            for ind in offsprings_s:
                ind.fitness.values, ind_model = toolbox.evaluate(ind, mapper[s], best_coop)
                models.append((ind.fitness.values, ind_model))
            # Replace the old population by the offspring
            pop_coop[s] = offsprings_s
            best_coop[s] = tools.selBest(pop_coop[s], 1)[0]
            best_gen_loss, best_gen_model = min(models)
            res_dict[(n,'loss')].append(best_gen_loss)
            
            if best_model_loss > best_gen_loss[0]:
                best_model = best_gen_model
                best_model_loss = best_gen_loss[0]
            
            if online_learning_AF:
                de.update_model_AF(layer_dict, best_coop[s], mapper[s])
            if online_learning_AF_WEIGHTS:
                model = best_gen_model
                de.model = model
                layer_dict = {}
                save_activation_location(model, layer_dict)
    best_model_test_acc = get_test_score(best_model, test_dataloader)
    last_model_test_acc = get_test_score(model, test_dataloader)
    res_dict[(n,'best_model')] = best_model
    res_dict[(n,'best_model_acc')] = best_model_test_acc
    res_dict[(n,'last_model')] = model
    res_dict[(n,'last_model_acc')] = last_model_test_acc 

np.save('results/'+name+'.npy', res_dict, allow_pickle=True)












