'''
Hyperparameter Tuning Plan

Values to be used for hyperparameter testing

'''

# Suggested Hyperparameter values for tuning

# Model Architecture Hyperparameters

#1
layer_vals = [8, 10, 12, 14, 16] 

#2
attention_head_vals = [8, 10, 12, 14, 16] 

# Training Hyperparameters

#3
learning_rate_vals = [9e-4, 1e-3, 2e-3]

# Optimizer Setting Hyperparameters

#4
weight_decay_vals = [1e-4, 1e-3, 0.01, 0.1, 0]