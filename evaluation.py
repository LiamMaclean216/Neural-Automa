import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.distributions import normal
import time
from utils import *
from rule import *
from IPython.display import clear_output

def evaluate_rule(rule, n_in, n_out, params, grid = None, give_labels = True):
    predicted_history = []
    label_history = []
    
    if grid is None:
        grid = rule.build_grid()#params["n_layers"],params["layer_width"])

    reward = 0
    for t in range(params["time_per_sample"]):
        #if(params["time_per_sample"] - t <= 15):
        
        
        if(not give_labels or t <= params["time_per_sample"] // 2):
            grid = insert_io(grid, torch.tensor(n_in), torch.tensor([0]))
        else:
            grid = insert_io(grid, torch.tensor(n_in), torch.tensor(n_out))
            
        if(params["verbose"]):
            clear_output(wait=True)
            pretty_print(grid)
        
        grid = rule.apply_rule(grid)
            
        n_predicted = grid[3][5][0]
        
        if(params["verbose"]):
            print(n_predicted.squeeze().detach().numpy().item())
            time.sleep(0.1)
        
        predicted_history.append(n_predicted.squeeze().detach().numpy().item())
        label_history.append(n_out[0])
        
        if t >= 2 and t <= params["time_per_sample"] // 2:
            reward += -torch.abs(torch.tensor(n_out) - n_predicted)
            #print(n_in, n_out, n_predicted)
            
    return reward.detach().numpy(), {"predicted_history" : predicted_history, "label_history" : label_history}

def evaluate_rule_on_generator(rule, gen, params):
    reward = 0
    grid = rule.build_grid()
    history = {"predicted_history" : [], "label_history" : []}
    
    give_labels = True
    for s in range(params["n_samples_per_evaluation"]):
        #if(s >= (params["n_samples_per_evaluation"]*4)//5):
        #    give_labels = False
            
        n_in, n_out = next(gen)
        r, h = evaluate_rule(rule, n_in, n_out, params, grid, give_labels)
        history["predicted_history"]+=(h["predicted_history"])
        history["label_history"]+=(h["label_history"])
        
        reward += r
        #history.append(h)
        
    return reward, history

def evaluate_population(rules, gen, params):
    history = [0] * params["population_size"]
    rewards = np.array([0] * params["population_size"])
    for idx, r  in enumerate(rules):
        rewards[idx], history[idx] = (evaluate_rule_on_generator(r, gen, params))
        
    arg_rewards = np.argsort(rewards)[::-1]
    #print(rewards)
    new_rules = []
    for r in arg_rewards[0:5]:
        for i in range(4):
            new_rules.append(rules[r].gen_child())
            
    return new_rules, {"best_rule" : rules[arg_rewards[0]], 
                       "best_reward" : rewards[arg_rewards[0]], "best_history" : history[arg_rewards[0]]}