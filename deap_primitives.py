import torch.nn as nn
import torch
from deap import gp


def protectedDiv(dividend, divisor):
    try:
        x=torch.div(dividend, divisor)
        if torch.isinf(x).item() or torch.isnan(x).item():
            return 1
        return x
    except:
        print('exception in protected div')
        return 1

def get_primitive_set():
    pset = gp.PrimitiveSet("MAIN", 1)
    pset.addPrimitive(torch.max, 2, name="max")
    pset.addPrimitive(torch.min, 2, name="min")
    pset.addPrimitive(torch.add, 2, name="add")
    pset.addPrimitive(torch.sub, 2, name="sub")
    pset.addPrimitive(torch.mul, 2, name="mul")
    pset.addPrimitive(protectedDiv, 2, name="protectedDiv")
    pset.addPrimitive(torch.tanh, 2, name="tanh")
    pset.addPrimitive(torch.ReLU(), 1, name="ReLU")
    pset.addPrimitive(torch.LeakyReLU(), 1, name="LeakyReLU")
    pset.addPrimitive(torch.ELU(), 1, name="ELU")
    pset.addPrimitive(torch.Hardshrink(), 1, name="Hardshrink")
    pset.addPrimitive(torch.CELU(), 1, name="CELU")
    pset.addPrimitive(torch.Hardtanh(), 1, name="Hardtanh")
    pset.addPrimitive(torch.Hardswish(), 1, name="Hardswish")
    pset.addPrimitive(torch.Softshrink(), 1, name="Softshrink")
    pset.addPrimitive(torch.RReLU(), 1, name="RReLU")
    # pset.addEphemeralConstant("rand101", lambda: random.randint(-1,1)) # check what this is
    pset.renameArguments(ARG0='x')
    return pset
