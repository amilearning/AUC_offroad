import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from itertools import product
from dataclasses import dataclass, field
# Define the grid dimensions
import time

def gen_img_set(batch = 100, grid_width= 64, grid_height=64, num_time_steps=10, show_sample_data = False):
    
    residual_error = np.random.randn(batch,num_time_steps,2)  # [x_delta, y_delta]
    for i in range(num_time_steps):
        residual_error[:,i,:] = residual_error[:,i,:]*(i+1)*0.25
    
            
    # Generate fake vehicle state and predicted actions
    fake_vehicle_state = np.random.randn(batch,5)  # [Vx, Vy, Wz, roll, pitch]
    fake_predicted_actions = np.random.rand(batch, num_time_steps, 2)  # [speed, steering angle] over 10 time steps
    fake_predicted_actions[:,:, 0] = (fake_predicted_actions[:,:, 0] * grid_width).astype(int)
    fake_predicted_actions[:,:, 1] = (fake_predicted_actions[:,:, 1] * grid_height).astype(int)
    
    # Initialize the output data with near-zero values
    output_data = np.full((batch, grid_width, grid_height, num_time_steps), 0.001) + np.random.rand(batch,grid_width, grid_height, num_time_steps)*0.1
    original_output = output_data[:,:,:,0].copy()
    std = 3

    rand_x_idx_mean = fake_predicted_actions[:,:,0] 
    rand_y_idx_mean = fake_predicted_actions[:,:,1] 

    rand_x_idx = np.linspace(rand_x_idx_mean-std,rand_x_idx_mean+std,2*std+1,axis=2).astype(int)
    rand_y_idx = np.linspace(rand_y_idx_mean-std,rand_y_idx_mean+std,2*std+1,axis=2).astype(int)
    batch_unique_combinations =  []
    for i in range(batch):        
        unique_combinations = list(product(np.transpose(rand_x_idx[i,:,:]), np.transpose(rand_y_idx[i,:,:])))
        unique_combinations = np.array(unique_combinations).astype(int)
        batch_unique_combinations.append(unique_combinations.copy())
        
    batch_unique_combinations = np.array(batch_unique_combinations)    
    batch_unique_combinations[:,:,0,:] = np.clip(batch_unique_combinations[:,:,0,:],0,grid_width-1)
    batch_unique_combinations[:,:,1,:] = np.clip(batch_unique_combinations[:,:,1,:],0,grid_height-1)
    
    # # Set the values in the output_data array
    for t in range(num_time_steps):
        for b in range(batch):
            tmp = output_data[b,batch_unique_combinations[b,:,0,t],batch_unique_combinations[b,:,1,t],t]
            rand_tmp = np.random.rand(*tmp.shape) * 0.7 + 0.3    
            output_data[b,batch_unique_combinations[b,:,0,t],batch_unique_combinations[b,:,1,t],t] = rand_tmp

  
    state_input = fake_vehicle_state
    action_input = fake_predicted_actions    
    image_input = original_output
    image_output = output_data
    
    if show_sample_data:
        time_step_to_visualize = 0
        fig, axs = plt.subplots(2, 2, figsize=(16, 12))
        # Plot image_output
        axs[0,0].imshow(image_output[0, :, :, time_step_to_visualize], cmap='jet', origin='lower')
        axs[0,0].set_title(f"Attention appleid Output at Time Step {time_step_to_visualize}")
        axs[0,0].set_xlabel("X")
        axs[0,0].set_ylabel("Y")

        # Plot original fake_grid
        axs[1,0].imshow(image_input[0, :, :, time_step_to_visualize], cmap='jet', origin='lower')
        axs[1,0].set_title("Original Fake Grid")
        axs[1,0].set_xlabel("X")
        axs[1,0].set_ylabel("Y")



        # Plot bar graph for state_input
        state_labels = ['Vx', 'Vy', 'Wz', 'Roll', 'Pitch']
        state_means = state_input[0,:]
        axs[0,1].bar(state_labels, state_means)
        axs[0,1].set_title("State Input")
        axs[0,1].set_xlabel("State")
        axs[0,1].set_ylabel("Mean Value")

        # Plot bar graph for action_input
        action_labels = [f"Step {i}" for i in range(num_time_steps)]
        action_means = action_input[0, :, 0]  # Taking the mean of speed values
        axs[1,1].bar(action_labels, action_means)
        axs[1,1].set_title("Action Input (Speed)")
        axs[1,1].set_xlabel("Time Step")
        axs[1,1].set_ylabel("Mean Speed")


        # Show the subplots
        plt.tight_layout()
        plt.show()
    
    return state_input, action_input, image_input, image_output, residual_error

# num_time_steps_ = 10
# state_input, action_input, image_input, image_output = gen_img_set(batch = 100, grid_width= 64, grid_height=64, num_time_steps=num_time_steps_, show_sample_data = True)
# time_step_to_visualize = 0

