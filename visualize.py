#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 28 17:34:21 2021

@author: christian
"""
import numpy as np
import numpy.ma as ma
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm

#TODO list
#1. visualize trajectory path of agent
 
########### VISUALIZING OUTPUT PREDICTIONS ##########
# input: tensor of shape [seq_len, 7, 9, 9] or [seq_len, 7, 11, 11]
# output: plot of sequence of gridstates with predictions
def seq_prediction(output, sequence):
    seq_len = sequence.shape[0]
    for i in range(seq_len):
        gs_prediction(output[i], sequence[i])

# input: tensor of shape [4] containing prediction for next action
#        input_batch_tensor of shape [7, 9, 9] containing gridstate
# output: plot of gridstate with prediction
def gs_prediction(output, gridstate):
    num_gridobj = gridstate.shape[0]
    gridsize = gridstate.shape[1]
    agent_idx = get_agent_idx(gridstate)
    walls = (gridsize==11)
    adj_idx_list = adjacent_indices(agent_idx, walls) #list of indices of all four squares adjacent to agent position
    prediction_tensor = torch.zeros((gridsize, gridsize))
    counter = 0
    for idx in adj_idx_list:
        if idx != None:
            prediction_tensor[idx[0], idx[1]] = output[counter]
            counter += 1
    pred_mask = prediction_tensor > 0    
    prediction_tensor = torch.tensor(ma.masked_array(prediction_tensor.numpy(), mask=pred_mask))
    
    image = torch.zeros(gridsize, gridsize)
    # ever grid_obj except agent and green food is distractor
    # indices: 0=start, 1=agent, 2=green, 3=blue, 4=red, 5=orange, 6=walls
    for i in range(num_gridobj):
        grid_obj = gridstate[i]
        if i!=0:
            grid_obj = grid_obj.multiply(i+1)
        image = image.add(grid_obj)
        # check if agent is on the same location as green food object and prevent it from turning red
        if i<3 and torch.max(image).item()==5:
            flat_idx = np.argmax(image.numpy())
            idx = np.unravel_index(flat_idx, (gridsize, gridsize))
            image[idx[0], idx[1]] = 2 # encodes the agent on the colormap
        
    if 7 in image:
        cmap = mcolors.ListedColormap(["white", "yellow", "dimgrey", "green","blue", "red", "orange", "black"], N=8)
    else:
        cmap = mcolors.ListedColormap(["white", "yellow", "dimgrey", "green", "blue", "red", "orange", "black"], N=7)
    plt.subplot(1,2,2)
    plt.imshow(image, cmap=cmap, interpolation="none") # image ohne predictions, wie in gridstate()! außerdem cmap_wall
    plt.imshow(prediction_tensor, cmap=cm.get_cmap("binary"), alpha=0.8, interpolation="none")
    plt.colorbar(shrink=0.5)
    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    plt.show()
    
    return

# input: input_batch_tensor of shape [7, 9, 9] containing gridstate 
# output: index of agent position as tuple
def get_agent_idx(gridstate):
    gridsize = gridstate.shape[1]
    flat_idx = np.where(gridstate[1].numpy().flatten() > 0)
    i, j = np.unravel_index(flat_idx, (gridsize, gridsize)) #i, j are 1x1 np-arrays
    agent_idx = (i.item(), j.item())
    return agent_idx

# input: tuple of agent position in 2D gridworld
#        boolean "walls" that is true if gridworld is 9x9
# output: index of all four squares adjacent to agent as list of tuples [(UP), (RIGHT), (DOWN), (LEFT)]
def adjacent_indices(agent_idx, walls):
    UP = (agent_idx[0]-1, agent_idx[1]) #implement check for out of bounds error in case agent is at the edge of 9x9 gridworld?
    RIGHT = (agent_idx[0], agent_idx[1]+1)
    DOWN = (agent_idx[0]+1, agent_idx[1])
    LEFT = (agent_idx[0], agent_idx[1]-1)
    if not walls:
        adj_idx_list = check_edge(agent_idx, [UP, RIGHT, DOWN, LEFT])
    else:
        adj_idx_list = [UP, RIGHT, DOWN, LEFT]
    return adj_idx_list

# currently only for 9x9 gridworld, probably won't be needed in 11x11
# input: input_batch_tensor of shape [7, 9, 9] containing gridstate
# output: list of adjacent indices as tuples, avoiding squares outside of the gridworld if agent is at the edge
def check_edge(agent_idx, adj_idx_list):
    # agent at top edge, delete UP
    if agent_idx[0] == 0:
        adj_idx_list[0] = None
    # agent at right edge, delete RIGHT
    if agent_idx[1] == 8:
        adj_idx_list[1] = None
    # agent at bottom edge, delete DOWN
    if agent_idx[0] == 8:
        adj_idx_list[2] = None
    # agent at left edge, delete LEFT
    if agent_idx[1] == 0:
        adj_idx_list[3] = None
        
    return adj_idx_list

########### VISUALIZATION FUNCTIONS FOR DEBUGGING ###########
# input: tensor of shape [7, 9, 9] or [7, 11, 11] if walls==True
# output: plot of gridstate
def gridstate(gridstate):
    num_gridobj = gridstate.shape[0]
    gridsize = gridstate.shape[1]
    image = torch.zeros(gridsize, gridsize)
    cmap_wall = mcolors.ListedColormap(["white", "yellow", "dimgrey", "green","blue", "red", "orange", "black"], N=8)
    cmap_nowall = mcolors.ListedColormap(["white", "yellow", "dimgrey", "green", "blue", "red", "orange", "black"], N=7)
    for i in range(num_gridobj):
        grid_obj = gridstate[i]
        if i!=0:
            grid_obj = grid_obj.multiply(i+1)
        image = image.add(grid_obj)
        
        # check if agent is on the same location as green food object
        if i<3 and torch.max(image).item()==5:
            flat_idx = np.argmax(image.numpy())
            idx = np.unravel_index(flat_idx, (gridsize, gridsize))
            image[idx[0], idx[1]] = 2 # encodes the agent on the colormap
            
    plt.figure()
    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    if 7 in image:
        plt.imshow(image, cmap_wall)
    else:
        plt.imshow(image, cmap_nowall)
        
# input: tensor of shape [seq_len, 7, 9, 9]
# output: plot of sequence of gridstates
def sequence(sequence):
    for step in sequence:
        gridstate(step)
        
### Path to text file ###
training_data = "data/list.txt"


########### VISUALIZATION FUNCTIONS TO CHECK GRIDWORLD DATA ###########
# functionality: read text file
# input: path to .txt file (type: string)
# output: data as list of strings, each string is one line of the text file (30 characters)
def read_data(path):
    with open(path, "r") as my_file:
        text = my_file.read()
        epochs = text.count("s") # counts number of trajectories, maybe useful?
        text = text.replace("s", "")
        data = text.splitlines()[1:]
        # remove empty lines
        for i in data:
            if i=="":
                data.remove(i)
    return data

# functionality: convert list of strings to list of 2D representations
# input: data as list of strings
# output: numpy array "twod_representation" of one gridworld-state
def step(data):
    step=data
    object_count = 0
    line_count = 0
    grid_state = np.zeros((7,9,9), dtype=int)
    for i in range(len(data)):
        ### for debugging ### 
        #print(i)
        #print("object: ", object_count)
        #print("line: ", line_count, "\n")
        if (line_count>8):
            line_count = 0
            object_count += 1 
        if (object_count > 6):
            object_count = 0
        if (len(data[i])>0):
            clean_string = data[i][3:28].replace(".", "").replace("s", "")
            line = list(clean_string.split()) #list of individual strings in one line
            line = list(map(int, line))
            grid_state[object_count][line_count] = line
            grid_state[object_count][line_count]
            line_count += 1
    twod_representation = np.zeros((9,9)) #dtype=int
    for i in range(7):
        grid_state[i] = grid_state[i]*(i+1)
        # black object (agent) moves and has to be treated differently
        twod_representation = np.add(twod_representation, grid_state[i])
    return twod_representation

# functionality: print each gridworld step
# input: path to .txt file (type: string)
# output:
def visualize_data(path):
    data_as_string = read_data(path)
    images = []
    cmap_wall = mcolors.ListedColormap(
    #weird: pink doesnt seem to matter?!
    ["white", "yellow", "black", "green", #"pink",
     "blue", "red", "orange", "saddlebrown"], N=8)
    cmap_nowall = mcolors.ListedColormap(
    #weird: pink doesnt seem to matter?!
    ["white", "yellow", "black", "green", #"pink",
     "blue", "red", "orange", "saddlebrown"], N=7)
    for i in range(0, len(data_as_string), 63):
        begin = i
        end = i + 63
        images.append(step(data_as_string[begin:end]))
    for im in images:
        if 7 in im:
            plt.figure()
            plt.imshow(im, cmap_wall)
        else:
            plt.figure()
            plt.imshow(im, cmap_nowall)
        print(im)
    
# function call
#visualize_data(training_data)    
