

import torch

''' 
Traverse through the model and put in layer_dict the sub-models, name of activation function 
and the original activation funtion.
Setting the activation in a specific layer can be done by:

layer_dict = {}
save_activation_location(model, layer_dict)
setattr(layer_dict[0][0], layer_dict[0][1], nn.SELU())

# layer_dict[0][2] will contain the original activation function

'''
def save_activation_location(model, layer_dict):
    for child_name, child in model.named_children():
        if child._get_name() in dir(torch.nn.modules.activation):
            layer_dict[len(layer_dict)] = [model, child_name, child]
        else:
            save_activation_location(child, layer_dict)
