#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 28 12:29:43 2021

@author: christian
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# plot running average of loss per epoch

def loss(epoch_train_loss, epoch_val_loss, test_loss, smooth=True):
    EPOCHS = len(epoch_train_loss)
    for i in range(EPOCHS):
        if not smooth:
            x=np.array(epoch_train_loss[i])
        else:
            x=pd.DataFrame(epoch_train_loss[i]).rolling(window=10).mean()
            x[0:9]=pd.DataFrame(epoch_train_loss[i][0:9])
            
        fig=plt.figure(figsize=(10,8))
        plt.plot(range(1,len(x)+1),x, label='Training Loss')
        plt.axhline(np.mean(epoch_val_loss[i]), linestyle='--', color='r',label='Validation Loss')
        if i==EPOCHS-1:
            plt.axhline(np.mean(test_loss), linestyle='--', color="g", label="Test Loss")
            
        plt.xlabel('batches')
        plt.ylabel('mean squared error loss')
        plt.ylim(0, 0.2) # consistent scale
        plt.xlim(0, len(x)+1) # consistent scale
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.title("Epoch " + str(i+1)) #+ str(hidden_dim))
        plt.show()

# plot running average accuracy per epoch
def acc(epoch_train_acc, epoch_val_acc, test_acc, smooth=True):
    EPOCHS = len(epoch_train_acc)
    for i in range(EPOCHS):
        if not smooth:
            x=np.array(epoch_train_acc[i])
        else:
            x=pd.DataFrame(epoch_train_acc[i]).rolling(window=10).mean()
            x[0:9]=pd.DataFrame(epoch_train_acc[i][0:9])
        fig=plt.figure(figsize=(10,8))
        plt.plot(range(1,len(x)+1),x, label='Training Accuracy')
        plt.axhline(np.mean(epoch_val_acc[i]), linestyle='--', color='r',label='Validation Accuracy')
        if i==EPOCHS-1:
            plt.axhline(np.mean(test_acc), linestyle='--', color="g", label="Test Accuracy")
            
        plt.xlabel('batches')
        plt.ylabel('accuracy')
        plt.ylim(0, 1) # consistent scale
        plt.xlim(0, len(x)+1) # consistent scale
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.title("Epoch " + str(i+1)) #+ str(hidden_dim))
        plt.show()