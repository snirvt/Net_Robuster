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


train_dataloader, val_dataloader = CIFAR10Data().train_val_dataloader()
test_dataloader = CIFAR10Data().test_dataloader()

inputs, classes = next(iter(train_dataloader))  
model = ConvNet_CIFAR10(inputs[0].shape)

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

pset = get_primitive_set()

de = DeapEvolution(model, pset, train_dataloader, val_dataloader)


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
ELITISM = 0
N_RUNS = 1
POP_SIZE = 1
NUM_SPECIES = 3
NGEN = 1
CXPB = 0.4
MUTPB = 0.4
CREATE_ORIGINAL_AF_IND=True
online_learning_AF = False
online_learning_AF_WEIGHTS = True
epsilon = 0.2
name = 'CIFAR10_coop_{}_layers_eps_{}_from_0'.format(NUM_SPECIES, epsilon)

debug = False

try:
    res_dict = np.load('results/'+name+'.npy', allow_pickle=True).item()
except:
    res_dict = {}
    np.save('results/'+name+'.npy',res_dict, allow_pickle=True)
for n in range(N_RUNS):
    pop_coop = []
    best_coop = []
    model = ConvNet_CIFAR10(inputs[0].shape)
    de = DeapEvolution(model, pset, train_dataloader, val_dataloader, epsilon)
    layer_dict = {}
    save_activation_location(model, layer_dict)

    for s in range(NUM_SPECIES):
        pop_coop.append(toolbox.population(n=POP_SIZE))
        if CREATE_ORIGINAL_AF_IND:
            for layer in mapper[s]:
                pop_coop[s].append(create_root_tree(layer_dict[layer][-1])[0])
        for ind in pop_coop[s]:
            ind.fitness.values, _ = toolbox.evaluate(model, ind, mapper[s], debug)
        best_coop.append(tools.selBest(pop_coop[s], 1)[0]) # make first gen equal original 

    best_model = deepcopy(model)
    best_model_coop = deepcopy(best_coop)
    best_model_loss = float('inf')
    res_dict[(n,'loss')] = []

    for g in range(1, NGEN):
        for s in range(NUM_SPECIES):
            # Select and clone the offspring
            offsprings_s = toolbox.select(pop_coop[s], len(pop_coop[s]))
            offsprings_s = [toolbox.clone(ind) for ind in offsprings_s]        
            offsprings_s = algorithms.varAnd(offsprings_s, toolbox, cxpb = CXPB, mutpb = MUTPB)
            
            if ELITISM > 0:
                offsprings_s[:ELITISM] = tools.selBest(pop_coop[s], ELITISM)[:ELITISM]
            
            gen_models = []
            gen_losses = []
            # print(model)
            for ind in offsprings_s:
                ind.fitness.values, ind_model = toolbox.evaluate(model, ind, mapper[s], debug)
                gen_losses.append(ind.fitness.values)
                gen_models.append(ind_model)
            # Replace the old population by the offspring
            pop_coop[s] = offsprings_s
            best_coop[s] = tools.selBest(pop_coop[s], 1)[0]
            best_model_idx = np.argmin(gen_losses)
            best_gen_loss = gen_losses[best_model_idx]
            best_gen_model = gen_models[best_model_idx]
            res_dict[(n,'loss')].append(best_gen_loss)
            
            if best_model_loss > best_gen_loss[0]:
                best_model = best_gen_model
                best_model_loss = best_gen_loss[0]
                best_model_coop = deepcopy(best_coop)
            if online_learning_AF:
                de.update_model_AF(layer_dict, best_coop[s], mapper[s])
            if online_learning_AF_WEIGHTS:
                model = best_gen_model
                de.set_model(model)
                layer_dict = {}
                save_activation_location(model, layer_dict)
    best_model_test_acc = get_test_score(best_model, test_dataloader, epsilon)
    last_model_test_acc = get_test_score(model, test_dataloader, epsilon)
    res_dict[(n,'best_coop')] = best_model_coop
    res_dict[(n,'best_coop_acc')] = best_model_test_acc
    res_dict[(n,'last_coop')] =  best_coop
    res_dict[(n,'last_coop_acc')] = last_model_test_acc 
    res_dict[(n,'best_model_str')] = str(best_model)
    res_dict[(n,'last_model_str')] = str(model)
    torch.save(model.state_dict(), 'results/'+name+'_{}_last_model_weights.pt'.format(n))
    torch.save(best_model.state_dict(), 'results/'+name+'_{}_best_model_weights.pt'.format(n))
    np.save('results/'+name+'.npy', res_dict, allow_pickle=True)

# np.save('results/'+name+'.npy', res_dict, allow_pickle=True)
# torch.save(model.state_dict(), 'results/'+name+'_last_model_weights.pt')
# torch.save(best_model.state_dict(), 'results/'+name+'_best_model_weights.pt')

# load_np = np.load('results/'+name+'.npy', allow_pickle=True).item()
# loaded_model = ConvNet_CIFAR10(inputs[0].shape)
# loaded_model.load_state_dict(torch.load('results/'+name+'_last_model_weights.pt'))
# loaded_model.eval()


# reduce population size, gnerations
# after found best model, try regular training on it
# compare to 1 level tree model
# compare to random evolution








