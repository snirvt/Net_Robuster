

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
        activation_options = dir(torch.nn.modules.activation) + ['ACTIVATION_MODULE', 'MAX', 'MIN', 'ADD', 'SUB', 'MUL', 'DIV', 'POW', 'LOG', 'COS', 'SIN']
        split_child_name = child._get_name().replace(')','(').split('(')
        if child._get_name() in activation_options or any(i in activation_options for i in split_child_name if i !=''):
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
        activation_module.__name__ = 'ACTIVATION_MODULE'
        # activation_module.type = str(self.activation)
        
    def __str__(self):
        return str(self.activation)
    def __call__(self, x):
        return self.activation(x)


def DyanmicNameActivationClass(class_name):
    class_str = '''class {}(torch.nn.Module):
                        \n\tdef __init__(self,activation):
                                \n\t\tsuper().__init__()
                                \n\t\tself.activation = activation
                                \n\t\tself.__name__ = str(self.activation)
                                \n\t\tself.type = str(self.activation)
                        \n\tdef __str__(self):
                            \n\t\treturn str(self.activation)
                        \n\tdef __call__(self, x):
                            \n\t\treturn self.activation(x)
                        '''
    exec(class_str.format(class_name))
    return eval('%s' % class_name)


def create_root_tree(func):
    pset = gp.PrimitiveSet("MAIN", 1)
    pset.addPrimitive(func, 1, name=str(func).replace('(','').replace(')','').upper())
    pset.renameArguments(ARG0='x')
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)
    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=1)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)
    return toolbox.population(n=1)


def single_layer_mapper(layer_dict):
    mapper = {0: range(0,len(layer_dict))}
    return mapper

def triple_mapper(layer_dict):
    mapper = {0: range(0,1),
              1:range(1,len(layer_dict)-1),
              2:range(len(layer_dict)-1, len(layer_dict))}
    return mapper

def each_mapper(layer_dict):
    mapper = {}
    for i in layer_dict.keys():
        mapper[i] = range(i,i+1)
    return mapper