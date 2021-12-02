import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.distributions import normal

class Rule(nn.Module):
    def __init__(self, params, mutation_rate = None):
        super(Rule, self).__init__()
        
        self.mutation_rate = mutation_rate
        
        if mutation_rate is None:
            self.mutation_rate = params["mutation_rate"]
        
        
        self.mutation_distribution = normal.Normal(0, self.mutation_rate)
        self.cell_state_size = params["cell_state_size"]
        self.params = params
        self.fc1 = nn.Linear(self.cell_state_size * 9 + len(params["hyperparameters"]), 4, bias = True)
        self.fc2 = nn.Linear(4, 4, bias = True)
        self.fc3 = nn.Linear(4, self.cell_state_size, bias = True)
        
    def forward(self,x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = torch.tanh(x)
        return x#.squeeze(-1)

    #get model parameters as vector
    def get_params(self):
        params = []
        for p in self.parameters():
            view = p.view(p.numel())
            params.append(view)
        params = torch.cat(params, dim=0)
        return params
    
        #turn vector into model parameters
    def set_params(self,data):
        idx = 0
        for p in self.parameters():
            view = data[idx:idx+p.numel()].view(p.shape)
            p.data = view
            idx+=p.numel()
        #return model
    
    def gen_child(self):
        mom = self.get_params()
        
        child = Rule(self.params, mutation_rate = self.mutation_rate * 0.995)
        child.set_params(mom + self.mutation_distribution.sample(mom.shape))
        
        return child

    def pad_cell_grid(self, x):
       # print(x)
        x_padded = torch.stack(
            [F.pad(x[:,:,i],(1,1,1,1), "constant", 0).unsqueeze(-1) for i in range(self.cell_state_size)]
        ,-1).squeeze(-2)
        
        return x_padded
    
    def apply_rule(self, x):
        x = x.unfold(0,3,1).unfold(1,3,1)
        #print(x)
        x = x.reshape([x.shape[0], x.shape[1], x.shape[2]* x.shape[3] * x.shape[4]])
        x = self.forward(x)
        #print(x.shape)
        
        x = self.pad_cell_grid(x)
        
        return x
    
    def build_grid(self):
        cells = torch.stack(
                [F.pad(torch.tanh(torch.rand(self.params["layer_width"], self.params["n_layers"])).float(),
                       (1,1,1,1), "constant", 0).unsqueeze(-1) for _ in range(self.cell_state_size)]
            ,-1).squeeze(-2)
        
        return cells