

import torch
import models.mobile_net as mobile_net

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


def get_pre_trained_model(device):
    model = mobile_net.mobilenet_v2(pretrained=True, device=device)
    return model


class activation_module(torch.nn.Module):
    def __init__(self, activation):
        super().__init__()
        self.activation = activation
   
    def __call__(self, x):
        return self.activation(x)

