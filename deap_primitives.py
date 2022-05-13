import random
import torch.nn as nn
import torch
from deap import gp


def protectedDiv(dividend, divisor):
    x = torch.div(dividend, divisor)
    x = torch.nan_to_num(x, nan=1.0, posinf=1.0, neginf=-1.0)
    return x
def get_primitive_set():
    pset = gp.PrimitiveSet("MAIN", 1)
    pset.addPrimitive(torch.max, 2, name="max")
    pset.addPrimitive(torch.min, 2, name="min")
    pset.addPrimitive(torch.add, 2, name="add")
    pset.addPrimitive(torch.sub, 2, name="sub")
    pset.addPrimitive(torch.mul, 2, name="mul")
    pset.addPrimitive(protectedDiv, 2, name="protectedDiv")
    pset.addPrimitive(torch.tanh, 1, name="tanh")
    pset.addPrimitive(nn.ReLU(), 1, name="ReLU")
    pset.addPrimitive(nn.LeakyReLU(), 1, name="LeakyReLU")
    pset.addPrimitive(nn.ELU(), 1, name="ELU")
    pset.addPrimitive(nn.Hardshrink(), 1, name="Hardshrink")
    pset.addPrimitive(nn.CELU(), 1, name="CELU")
    pset.addPrimitive(nn.Hardtanh(), 1, name="Hardtanh")
    pset.addPrimitive(nn.Hardswish(), 1, name="Hardswish")
    pset.addPrimitive(nn.Softshrink(), 1, name="Softshrink")
    pset.addPrimitive(nn.RReLU(), 1, name="RReLU")
    pset.addEphemeralConstant("rand101", lambda: random.randint(-1,1))
    pset.renameArguments(ARG0='x')
    return pset
