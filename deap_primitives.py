import random
import torch.nn as nn
import torch
from deap import gp

from torch.nn import Hardtanh

def protected_div(dividend, divisor):
    x = torch.div(dividend, divisor)
    x = torch.nan_to_num(x, nan=1.0, posinf=1.0, neginf=-1.0)
    return x

def protected_pow(x,y):
    res = torch.pow(x,y)
    res = torch.nan_to_num(res, nan=1.0, posinf=1.0, neginf=-1.0)
    return res


def protected_log(x):
    x = torch.log(torch.abs(x))
    x = torch.nan_to_num(x, nan=1.0, posinf=1.0, neginf=-1.0)
    return x


def get_primitive_set():
    pset = gp.PrimitiveSet("MAIN", 1)
    pset.addPrimitive(torch.max, 2, name="MAX")
    pset.addPrimitive(torch.min, 2, name="MIN")
    pset.addPrimitive(torch.add, 2, name="ADD")
    pset.addPrimitive(torch.sub, 2, name="SUB")
    pset.addPrimitive(torch.mul, 2, name="MUL")
    # pset.addPrimitive(torch.pow, 2, name="POW")
    # pset.addPrimitive(torch.sin, 1, name="SIN")
    # pset.addPrimitive(torch.cos, 1, name="COS")
    # pset.addPrimitive(protected_log, 1, name="LOG")
    # pset.addPrimitive(protected_div, 2, name="DIV")
    pset.addPrimitive(torch.tanh, 1, name="TANH")
    pset.addPrimitive(nn.ReLU(), 1, name="RELU")
    pset.addPrimitive(nn.LeakyReLU(), 1, name="LEAKYRELU")
    pset.addPrimitive(nn.ELU(), 1, name="ELU")
    pset.addPrimitive(nn.Hardshrink(), 1, name="HARDSHRINK")
    pset.addPrimitive(nn.CELU(), 1, name="CELU")
    pset.addPrimitive(nn.Hardtanh(), 1, name="HARDTANH")
    pset.addPrimitive(nn.Hardswish(), 1, name="HARDWISH")
    pset.addPrimitive(nn.Softshrink(), 1, name="SOFTSHRINK")
    pset.addPrimitive(nn.RReLU(), 1, name="RRELU")
    # pset.addEphemeralConstant("RAND101", lambda: random.randint(-1,1))
    pset.addEphemeralConstant("RAND_UNF", lambda: random.uniform(-2,2))
    pset.renameArguments(ARG0='x')
    return pset
