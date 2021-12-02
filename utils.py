import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.distributions import normal

def gate_sample(gate = "XOR", batch_size = 1):
   # input = (np.random.rand(N, 2) > 0.5) * 1
    
    values = np.array([[0,0],[0,1], [1,0], [1,1]])
    np.random.shuffle(values)
    if gate == "XOR":
        out = (np.logical_xor(values[:,0], values[:,1]) * 1)
        
    elif gate == "AND":
        out = (np.logical_and(values[:,0], values[:,1]) * 1)
    
    out = np.expand_dims(out, axis=-1)
    return values*2-1, out*2-1

def gate_sample_gen(gate = "XOR"):
    while True:
        sample = gate_sample(gate = gate)
        for (i,o) in zip(sample[0], sample[1]):
            yield i,o

def pretty_print(a, cutoff = 13):
    with np.printoptions(formatter=dict(float=lambda t: "%5.3f" % t), precision=2):
        for i in a:
            for j in i:
                #vector_values = np.abs(j.detach().numpy()).__str__()
                vector_values = j.detach().numpy().__str__()
                if len(vector_values) > cutoff:
                    print(vector_values[:cutoff-4] + "...]",end="")
                    continue
                
                print(vector_values,end="")
            print("")
            
def insert_io(grid, i, o):
    grid[:, 0][1:3] = i.unsqueeze(-1)
    grid[:, -1][2] = o.unsqueeze(-1)
    return grid