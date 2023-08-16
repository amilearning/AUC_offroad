import numpy as np
import torch
import matplotlib.pyplot as plt
import os

np_ws = os.path.expanduser('~') + '/np_ws/'
test_result_dir = os.path.join(np_ws, 'test_result/')
model_dir = os.path.join(np_ws, 'models/')
ltatt_weight_dir = os.path.join(np_ws, 'models/ltatt.pth')


def get_context_target_data(state, action, image_in, residual_out):
    # random_integers = torch.randint(int(state.shape[0]/2), int(state.shape[0]*4/5), size=(1,)).cpu().numpy()[0]
    random_integers = int(state.shape[0]/2)+1
        
    context_state = state[:random_integers]
    context_action = action[:random_integers]
    context_image_in = image_in[:random_integers]        
    context_residual_out = residual_out[:random_integers]
    
    target_state = state[random_integers:]
    target_action = action[random_integers:]
    target_image_in = image_in[random_integers:]        
    target_residual_out = residual_out[random_integers:]

    context_data = (context_state, context_action, context_image_in, context_residual_out)
    target_data  = (target_state, target_action, target_image_in, target_residual_out)
    return context_data, target_data 
        

def animate_result(batch_images_np = None,outputs_np = None, epoch = 0):
        if torch.is_tensor(batch_images_np):
            batch_images_np = batch_images_np.cpu().numpy()
            
        if torch.is_tensor(outputs_np):
            outputs_np = outputs_np.cpu().numpy()     
            
        # Create a figure with subplots for each image pair
        fig, axes = plt.subplots(2, batch_images_np.shape[-1],figsize=( 4 * batch_images_np.shape[-1],8))
        
        for j in range(batch_images_np.shape[-1]):
            
            axes[0, j].imshow(batch_images_np[:,:,j].squeeze(), cmap='gray')
            axes[0, j].set_title('Ground Truth')
            
            axes[1, j].imshow(outputs_np[:,:,j].squeeze(), cmap='gray')
            axes[1, j].set_title('Predicted Output')
        
        plt.tight_layout()
        
                # Save the subplots in a specific directory
        
        if not os.path.exists(test_result_dir):
            os.makedirs(test_result_dir)
        
        plt.savefig(os.path.join(test_result_dir, f'evaluation_subplots_{epoch}.png'))

        
        plt.close(fig)