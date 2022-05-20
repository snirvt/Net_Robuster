

from pyparsing import ParseSyntaxException
import torch
import models.mobile_net as mobile_net
from deap import creator
from deap import gp
from deap import base
from deap import tools

''' 
Traverse through the model and put in layer_dict the sub-models, name of activation function 
and the original activation funtion.
Setting the activation in a specific layer can be done by:

layer_dict = {}
save_activation_location(model, layer_dict)
setattr(layer_dict[0][0], layer_dict[0][1], nn.SELU())

# layer_dict[0][2] will contain the original activation function

'''
# https://stackoverflow.com/questions/58297197/how-to-change-activation-layer-in-pytorch-pretrained-module
def save_activation_location(model, layer_dict):
    for child_name, child in model.named_children():
        if child._get_name() in dir(torch.nn.modules.activation):
            layer_dict[len(layer_dict)] = [model, child_name, child]
        else:
            save_activation_location(child, layer_dict)

def replace_model_activations(layer_dict, activation_list, activation_idx):
    for i in range(len(activation_list)):
        idx = activation_idx[i]
        setattr(layer_dict[idx][0], layer_dict[idx][1], activation_list[i])


def replace_model_activations_2(layer_dict, activation, range_layers):
    for i in range_layers:
        setattr(layer_dict[i][0], layer_dict[i][1], activation)


def get_pre_trained_model(device):
    model = mobile_net.mobilenet_v2(pretrained=True, device=device)
    return model


class activation_module(torch.nn.Module):
    def __init__(self, activation):
        super().__init__()
        self.activation = activation
   
    def __call__(self, x):
        return self.activation(x)


def create_root_tree(func):
    pset = gp.PrimitiveSet("MAIN", 1)
    pset.addPrimitive(func, 1, name=str(func).replace('(','').replace(')',''))
    pset.renameArguments(ARG0='x')
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)
    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=1)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)
    return toolbox.population(n=1)