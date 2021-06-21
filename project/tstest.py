"""TorchScript Test."""# coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2021, All Rights Reserved.
# ***
# ***    File Author: Dell, 2021年 06月 17日 星期四 14:09:56 CST
# ***
# ************************************************************************************/
#


import torch

import pdb

@torch.jit.script
def foo(len:int)->torch.Tensor:
    rv = torch.zeros(3, 4)
    for i in range(len):
        if i < 10:
            rv = rv - 1.0
        else:
            rv = rv + 1.0
    return rv

# print(foo.code)
# print(foo.graph)
# pdb.set_trace()

@torch.jit.script
def fn(x):
    # type: (torch.Tensor) -> torch.Tensor
    result = x[0]
    for i in range(x.size(0)):
        result = result * x[i]
    return result



class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.x = torch.randn(2, 3)

    def forward(self, h: int, w : int) -> torch.Tensor:
        return self.x * h * w

m = torch.jit.script(Model())

pdb.set_trace()

@torch.jit.script
def foo1(x, tup):
    # type: (int, Tuple[Tensor, Tensor]) -> Tensor
    t0, t1 = tup
    return t0 + t1 + x

y = (torch.rand(3), torch.rand(3))
print(foo1(3, y))



from typing import Dict, List, Tuple

class EmptyDataStructures(torch.nn.Module):
    def __init__(self):
        super(EmptyDataStructures, self).__init__()

    def forward(self, x: torch.Tensor) -> Tuple[List[Tuple[int, float]], Dict[str, int]]:
        # This annotates the list to be a `List[Tuple[int, float]]`
        my_list: List[Tuple[int, float]] = []
        for i in range(10):
            my_list.append((i, x.item()))

        my_dict: Dict[str, int] = {}
        return my_list, my_dict

e = torch.jit.script(EmptyDataStructures())



@torch.jit.script
class Pair:
  def __init__(self, first, second):
    self.first = first
    self.second = second

@torch.jit.script
def sum_pair(p):
  # type: (Pair) -> Tensor

  print("first: ", p.first)
  print("second: ", p.second)
  return p.first + p.second

p = Pair(torch.rand(2, 3), torch.rand(2, 3))
p.second = 10.0 + torch.randn(2, 3)

print(sum_pair(p))


class SubModule(torch.nn.Module):
    def __init__(self):
        super(SubModule, self).__init__()
        self.weight = torch.nn.Parameter(torch.randn(2))

    def forward(self, input):
        return self.weight + input

class MyModule(torch.nn.Module):
    __constants__ = ['mods']

    def __init__(self):
        super(MyModule, self).__init__()
        self.mods = torch.nn.ModuleList([SubModule() for i in range(10)])

    def forward(self, v):
        for module in self.mods:
            v = module(v)
        return v


m = torch.jit.script(MyModule())

pdb.set_trace()

