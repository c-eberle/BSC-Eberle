#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import random
import convlstm
import visualize
import plot_results

from progress import Progress as P

""" Initialization of parameters """
# check if GPU is available
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("running on GPU")
else:
    device = torch.device("cpu")
    print("running on CPU")

# global variables for NN
EPOCHS = 3
num_layers = 2
hidden_dim=[16, 16]
num_classes = 4
learning_rate = 0.001
batch_size = 64
kernel_size=(3,3)

# seeds used for optimizing the net
initial_seed = 0
random.seed(initial_seed)
torch.manual_seed(initial_seed)

# global variables for data
training_data = ["daten/DatenTeacher20k.txt", "daten/DatenTest20k.txt"]
validation_data = ["daten/DatenTeacher2k.txt", "daten/DatenTest2k.txt"]
testing_data = ["daten/DatenTeacher10k.txt", "daten/DatenTest10k.txt"]
print("Training data: " + training_data[0], training_data[1])
print("Validation data: " + validation_data[0], validation_data[1])
print("Testing data: " + testing_data[0], testing_data[1])
outside_walls = True
gridworld_side = 0
if outside_walls:
    gridworld_side = 11
else:
    gridworld_side = 9
gridworld_dim = gridworld_side*gridworld_side
num_gridobj = 6
print("-> gridworld dimensions are " + str(num_gridobj) + "x" + str(gridworld_side) +"x" + str(gridworld_side))


''' Helper Methods for processing of data'''

"""
functionality:  transforms teaching-data in tensors and creates the variable episode_length
input:          data: the data set as string, which gets transformed
                episodes: number of trajectories in the data set
output:         teach_data: transformed data-set as tensor
                episode_length: array, which shows the the time-steps of each episode
usage:          The output of the NN are tensors. To be able to compare the output of the NN 
                to the teaching-data, this data must also be in tensor-form. 
                'episode_length' is needed to know the dimensions of the tensors that 
                are being computed. As these dimensions are the same for the teacher data set and
                the tracjectory data set this variable can also be used in the function 
                'transform_to_input_tensor' and is thus part of the output
"""
def transform_to_output_tensor(data, episodes):
    # definition of output variables
    teach_data = [] 
    episode_length = np.zeros((episodes))

    counter = 0
    # caches the input for every episode
    episode_input = []

    # dat is a single-character string
    for dat in data:
        if dat == "s":
            episode_length[counter] = len(episode_input)
            teach_data.append(episode_input) # array wird initialisiert
            counter += 1
            episode_input = []
            continue
        else:
            # movement data of one time-step (4-dim one-hot vector)
            temp = [] 
            for elem in dat:
                if elem == "0" or elem == "1":
                    temp.append(int(elem))
                    
            episode_input.append(temp)
                
    # transforms every list entry to tensor
    teach_data = [torch.DoubleTensor(teaching) for teaching in teach_data]
    
    return teach_data, episode_length



"""
functionality:  transforms input-data to tensors
input:          data: the data set (as a string) which is to be transformed
                episode_length: list showing the length of each episode
output:         training_data: transformed data-set as tensor
usage:          The NN needs tensors as input.
"""
def transform_to_input_tensor(data, episode_length):
    # temporary Variables
    initial_entry = 8
    gs = 9 # has to be nine, if outside_walls then walls will be added later
    gd = 9*9
    go = 7

    # Definition of output variable
    training_data = [[] for l in range(len(episode_length))] 

    # transform training_data to correct dimension
    for i in range(len(episode_length)):
        training_data[i] = [[[[initial_entry for w in range(gs)] for x in range(gs)] for y in range(go)] for z in range(int(episode_length[i]))]
        
    # read data into training_data
    count = 0
    for episode, dat in enumerate(data):
        for elem in dat:
            if elem == "0" or elem == "1":
                if count//(go*gd) < len(training_data[episode]):
                    #if episode>143 and (count // (go*gd))>7:
                    #    print([episode],[count // (go*gd)],[(count % (go*gd)) // gd],[(count % gd) // gs],[count % gs])
                    training_data[episode][count // (go*gd)][(count % (go*gd)) // gd][(count % gd) // gs][count % gs] = int(elem)
                count += 1
                
        count = 0

    # transform training_data to list of tensors
    training_data = [torch.DoubleTensor(ls) for ls in training_data]

    return training_data


"""" 
functionality: create list of lists of batched input and teacher tensors
input: train_data_batch, a list of tensors, each tensor of shape [seq_len+1, 7, 9, 9]
       teach_data_batch, a list of tensors, each tensor of shape [seq_len, 4]
output: train_data_batch_list, a list of lists tensors, each list of length batch_size
        and each tensor of shape [seq_len, 7, 9, 9]
        teach_data_batch_list, a list of lists tensors, each list of length batch_size
        and each tensor of shape [seq_len, 4]
"""
def transform_to_batch_list(train_data_batch, teach_data_batch):
    num_batches = int(len(train_data_batch) / batch_size)
    input_batch_tensor_list = [0] * num_batches
    teach_batch_tensor_list = [0] * num_batches
    
    counter = 0
    for i in range(num_batches):
        input_batch_tensor_list[i] = torch.stack(train_data_batch[counter:counter+batch_size])
        teach_batch_tensor_list[i] = torch.stack(teach_data_batch[counter:counter+batch_size])
        counter += batch_size
        # clean input data: remove last gridstate and gridworld-dimension for start object
        seq_len = teach_batch_tensor_list[i][0].shape[0]
        input_batch_tensor_list[i] = input_batch_tensor_list[i][:, :seq_len, 1:7, :, :]
        
    return input_batch_tensor_list, teach_batch_tensor_list


"""
funcitonality:  sorts data into batches of given size, so that every batch only includes episodes 
                of the same length. Data is first sorted by length of each episode to fill the batches, 
                later the finished batches are shuffled again to avoid sorted input into the LSTM
input:          training_data (tensor)
                teaching_data (tensor)
                batch_size
output:         training_data_batched / teaching_data_batched: tensor of training / teaching data, 
                sorted into batches of episodes with the same number of timesteps. Batch structure is 
                not visible in tensor structure
usage:          LSTM needs batches in which every entry has the same dimension               
"""
def create_batches (training_data, teaching_data, batch_size):
    # zipped_data combines the training and teaching datasets into one dataset sorted by length
    zipped_data = [[training_data[i], teaching_data[i]] for i in range(len(training_data))]
    zipped_data.sort(key = sort_by_length)
    zipped_data_batched = []
    
    while zipped_data != []:
        
        current_observed_length = len(zipped_data[0][0])
        current_batch = []
        for i in range(batch_size):
            # add next element to current batch, if it has the same length
            if zipped_data != [] and len(zipped_data[0][0]) == current_observed_length:
                next_element = zipped_data.pop(0)
                current_batch.append(next_element)
            # current batch can't be completed (because the next element doesn't have the same length) 
            # and thus gets deleted
            else:
                current_batch = []
                break
        
        # current batch is added to zipped_data (unless it is empty)
        if current_batch != []:
            zipped_data_batched.append(current_batch)
            
    random.shuffle(zipped_data_batched)
    
    # deletes batch structure
    zipped_data_batched_flattened = flatten(zipped_data_batched, batch_size)
    
    # unzips data
    training_data_batched = [zipped_data_batched_flattened[i][0] for i in range(len(zipped_data_batched_flattened))]
    teaching_data_batched = [zipped_data_batched_flattened[i][1] for i in range(len(zipped_data_batched_flattened))]
        
    return training_data_batched, teaching_data_batched


"""helper function for sorting the episodes"""
def sort_by_length(val):
    return len(val[0])


"""helper function that reduces a list of tensors by one dimension"""
def flatten(list_of_tensors, batch_size):
    flat_tensor = [0] * len(list_of_tensors) * batch_size
    counter = 0
    for i in range(len(list_of_tensors)):
        for j in range(len(list_of_tensors[i])):
            flat_tensor[counter] = list_of_tensors[i][j]
            counter += 1
    return flat_tensor


"""
functionality: Add outside walls to the gridworld, turning it from a 9x9 to a 11x11 gridworld
input: train_data_list - a list of all sequences from the dataset
output: train_data_list - the same list of all sequences with added outside walls
"""
def add_outside_walls(input_batch_tensor):
    seq_len = input_batch_tensor.shape[1]
    num_gridobject = 6 #remove once num_gridobj=6
    out_tensor = torch.zeros(batch_size, seq_len, num_gridobj, 11, 11)
    
    #pad with zeros for non-wall objects
    out_tensor[:, :, 0:num_gridobject-1, :, :] = F.pad(input_batch_tensor[:, :, 0:num_gridobject-1, :, :], (1,1,1,1), "constant", 0)
    #pad with ones for wall-object
    out_tensor[:, :, num_gridobject-1, :, :] = F.pad(input_batch_tensor[:, :, num_gridobject-1, :, :], (1,1,1,1), "constant", 1) 
    return  out_tensor



"""
Data input: reads and transforms data to batched tensors
"""

"""training data"""
with open(training_data[0], "r") as my_file:
    text = my_file.read()
    train_episode_number = text.count("s")
    data = text.splitlines()
    data.remove("")
    
    teach_data, train_episode_length = transform_to_output_tensor(data, train_episode_number)
    
with open(training_data[1], "r") as my_file:
    data = my_file.read().split("s")
    data.remove("")
    for i in range(len(train_episode_length)):
       train_episode_length[i] += 1
       
    train_data = transform_to_input_tensor(data, train_episode_length)
        
[train_data_batch, teach_data_batch] = create_batches(train_data, teach_data, batch_size)
[train_batch_list, teach_batch_list] = transform_to_batch_list(train_data_batch, teach_data_batch)

""" validation data """
with open(validation_data[0], "r") as my_file:
    text = my_file.read()
    val_episode_number = text.count("s")
    data1 = text.splitlines()
    data1.remove("")
    
    teach_data_val, val_episode_length = transform_to_output_tensor(data1, val_episode_number)
    
with open(validation_data[1], "r") as my_file:
    data1 = my_file.read().split("s")
    data1.remove("")
    for i in range(len(val_episode_length)):
       val_episode_length[i] += 1
       
    train_data_val = transform_to_input_tensor(data1, val_episode_length)
        
[train_data_batch_val, teach_data_batch_val] = create_batches(train_data_val, teach_data_val, batch_size)
[train_batch_list_val, teach_batch_list_val] = transform_to_batch_list(train_data_batch_val, teach_data_batch_val)

'''testing data'''
with open(testing_data[0], "r") as my_file:
    text = my_file.read()
    test_episode_number = text.count("s")
    data2 = text.splitlines()
    data2.remove("")
    
    teach_data_test, test_episode_length = transform_to_output_tensor(data2, test_episode_number)
    
with open(testing_data[1], "r") as my_file:
    data2 = my_file.read().split("s")
    data2.remove("")
    for i in range(len(test_episode_length)):
       test_episode_length[i] += 1
    
    train_data_test = transform_to_input_tensor(data2, test_episode_length)
    
[train_data_batch_test, teach_data_batch_test] = create_batches(train_data_test, teach_data_test, batch_size)
[train_batch_list_test, teach_batch_list_test] = transform_to_batch_list(train_data_batch_test, teach_data_batch_test)



"""Initialization of progress bar"""
"""progress.py needs to be in the same file and progress must be installed"""
p = P(int(len(train_data_batch)/batch_size), mode = 'bar')

progress_epoch = P.Elememt("Epoch",0) 
progress_batch = P.Elememt("Batch",0,display_name='hide',max_value = int(len(train_data_batch)/batch_size), value_display_mode = 1)
progress_time = P.ProgressTime(postfix="/epoch")
progress_loss = P.Elememt("Loss",0)
progress_acc = P.Elememt("Acc",0)

bar = P.Bar()

p = p(progress_epoch)(bar)(progress_batch)(progress_time)("- ")(progress_loss)("- ")(progress_acc) # format progress bar
p.get_format()

p.initialize()


""" Training the net"""

"""Initialization of ConvLSTM and optimizer"""
# Parameters: ConvLSTM(input_dim, hidden_dim, kernel_size, num_layers, batch_first, bias, return_all_layers, walls_on)
model = convlstm.ConvLSTM(num_gridobj, hidden_dim, kernel_size, num_layers, True, True, False, outside_walls).to(device)
optimizer = optim.Adam(model.parameters(), lr = 0.0001)
num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(model)
print("number of trainable parameters: " + str(num_parameters) + "  -  hidden dimension: " + str(hidden_dim))

# track accuracy and loss over all epochs
[epoch_train_acc, epoch_train_loss] = [[], []]
[epoch_val_acc, epoch_val_loss] = [[], []]

for epoch in range(EPOCHS):
    
    # to track the training and validation accuracy/loss   
    [train_acc, train_loss] = [[], []]
    [val_acc, val_loss] = [[], []]
    for i in range(len(train_batch_list)):
        # sets the gradient to zero before each learning phase
        model.train()
        model.zero_grad()
        optimizer.zero_grad()
        
        seq_len = teach_batch_list[i][0].shape[0]
        input_batch_tensor = add_outside_walls(train_batch_list[i])
        teach_batch_tensor = teach_batch_list[i].reshape(seq_len * batch_size, 4)
        
        # forward pass
        output = model.forward(input_batch_tensor.float()).double()

        # calculate loss
        loss = F.mse_loss(output, teach_batch_tensor)
        
        # backpropagation
        loss.backward()
        optimizer.step()
        
    
        # creates an array of booleans which is true, when the prediction of
        # the NN is equal to the real result
        #boolean_list = [(np.argmax(output[i].detach().numpy()) == np.argmax(teach_batch_tensor[i])) for i in range(output.shape[0])]
        boolean_list = [(torch.argmax(output[i]) == torch.argmax(teach_batch_tensor[i])) for i in range(output.shape[0])]
        acc = round(np.mean(boolean_list), 3)
        
        # shows the accuracy of each batch
        train_acc.append(acc)
        train_loss.append(loss.item())
        
        progress_loss(round(loss.item(), 4))
        progress_acc(round(np.mean(train_acc), 4))
        progress_batch(i+1)
        progress_epoch(epoch+1)
        p.update(step = 1)
        
    # setting progress bar to new line at the end of each epoch
    p.set_cursor_position()
    
    ####### validation #######
    model.eval()
    for i in range(len(train_batch_list_val)):
        seq_len = teach_batch_list_val[i][0].shape[0]
        val_input_batch_tensor = add_outside_walls(train_batch_list_val[i])
        val_teach_batch_tensor = teach_batch_list_val[i].reshape(seq_len * batch_size, 4)
        
        # forward pass
        output = model.forward(val_input_batch_tensor.float()).double()
        # calculate loss
        loss = F.mse_loss(output, val_teach_batch_tensor)
        # calculate accuracy
        boolean_list = [(torch.argmax(output[i]) == torch.argmax(val_teach_batch_tensor[i])) for i in range(output.shape[0])]
        acc = round(np.mean(boolean_list), 3)
        
        val_acc.append(acc)
        val_loss.append(loss.item())
    print("Validation loss: " + str(round(np.mean(val_loss), 4)) + 
          "   -   Validation accuracy: " + str(round(np.mean(val_acc), 4)))
    
    # save accuracies and losses of this epoch
    epoch_train_acc.append(train_acc)
    epoch_train_loss.append(train_loss)
    epoch_val_acc.append(val_acc)
    epoch_val_loss.append(val_loss)
    
    """ Model testing"""
"""
puts the testing data through the trained LSTM and computes the a
i.e. how often the LSTM guesses correctly
"""

correct = 0
total = 0

""" 
Variablen für die Visualisierung
"""
num_vis = 0 #Anzahl an Sequenzen die visualisiert werden sollen

[test_acc, test_loss] = [[], []]

with torch.no_grad():
    for i in range(len(train_batch_list_test)):
        seq_len = teach_batch_list_test[i][0].shape[0]
        test_input_batch_tensor = add_outside_walls(train_batch_list_test[i])
        test_teach_batch_tensor = teach_batch_list_test[i].reshape(seq_len * batch_size, 4)
        
        # forward pass
        output = model.forward(test_input_batch_tensor.view(batch_size, seq_len, num_gridobj, 11, 11).float()).double()
        
        # calculate loss
        loss = F.mse_loss(output, test_teach_batch_tensor)

        # creates an array of booleans which is true, when the prediction of
        # the NN is equal to the real result
        boolean_list = [(torch.argmax(output[i]) == torch.argmax(test_teach_batch_tensor[i])) for i in range(output.shape[0])]
        acc = round(boolean_list.count(True)/len(boolean_list), 3)
        
        # shows the accuracy of each batch
        test_acc.append(acc)
        test_loss.append(loss)
        
        
        # in the first num_vis batches, visualize the first sequence
        if i<num_vis:
            output = output.reshape(batch_size, seq_len, 4)
            if outside_walls:
                visualize.seq_prediction(output[0], add_outside_walls(test_input_batch_tensor)[0])
            else:
                visualize.seq_prediction(output[0], test_input_batch_tensor[0])
        
    print("### Out of sample testing ###")
    print("Loss: ", round(float(loss), 4), " - Accuracy: ", round(np.mean(test_acc), 4))

""" Cool data stats (optional) """
def avg_seq_len(train_data_batch):
    seq_len_list = []
    for seq in train_data_batch:
        seq_len = seq.shape[0]
        seq_len_list.append(seq_len)
    return np.average(seq_len_list)

seq_len_train = round(avg_seq_len(train_data_batch), 3)
seq_len_val = round(avg_seq_len(train_data_batch_val), 3)
seq_len_test = round(avg_seq_len(train_data_batch_test), 3)

#print("\n Average sequence lengths:")
#print("train data: " + str(seq_len_train) + "   val data: " + str(seq_len_val) + "   test data: " + str(seq_len_test))

""" Plot loss and accuracy """
plot_results.loss(epoch_train_loss, epoch_val_loss, test_loss, smooth=False)
plot_results.acc(epoch_train_acc, epoch_val_acc, test_acc, smooth=False)
plot_results.loss(epoch_train_loss, epoch_val_loss, test_loss, smooth=True)
plot_results.acc(epoch_train_acc, epoch_val_acc, test_acc, smooth=True)
