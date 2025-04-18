'''
Hyperparameter Tuning Plan

Below are some suggested values for tuning the model hyperparameters.  
Likely won't test all of these, but these are some values to test.

Full plan to come ....

'''

# Suggested Hyperparameter values for tuning

# Model Architecture Hyperparameters

layer_vals = [8, 10, 12, 14, 16] 

attention_head_vals = [8, 10, 12, 14, 16] 

# Training Hyperparameters

learning_rate_vals = [9e-4, 1e-3, 2e-3]

# Optimizer Setting Hyperparameters

weight_decay_vals = [1e-4, 1e-3, 0.01, 0.1, 0]