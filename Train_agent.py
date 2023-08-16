

import torch
import torch.nn as nn
from fake_data_gen import gen_img_set
from gat_utils import GATDataset
from torch.utils.data import DataLoader


from tensorboardX import SummaryWriter 
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from train_utils import *
import time
from modules.AUC_anprnn_model import AUCANPRNN_Model



class AUCTrainAgent():
    def __init__(self):
                
        self.writer = SummaryWriter()

        self.args = {'input_grid_width':64,
                'input_grid_height':64,
                'n_time_step':10, 
                'lstm_hidden_size': 12,  
                'init_fc_hidden_size':64,
                'input_state_dim':5, # [vx, vy, wz, roll, pitch] 
                'input_action_dim':2, # [vx, delta] 
                'data_size':1000,
                'batch_size':150,
                'num_epochs': 20,
                'output_residual_dim': 2
                }
        self.train_dataloader = None
        self.test_dataloader = None
        
        self.ltatt_model_path = None
        ## Generate train and test dataloader

        # Initialize model and move to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        
        
        ## define models for training
        # 1. local temporal attention module
        # self.ltatt_model = LocalTemporalAttention(args = args).to(device)
        
        # 2. AUC_ANP_RNN module 
        self.auc_anp_rnn_model = AUCANPRNN_Model(args= self.args).to(self.device)
        
        

    def gen_fake_dataloaders(self):
        args = self.args
        state_input, action_input, image_input, image_output, residual_error = gen_img_set(batch = args['data_size'], grid_width= args['input_grid_width'], grid_height=args['input_grid_height'], num_time_steps=args['n_time_step'], show_sample_data = False)
        test_state_input, test_action_input, test_image_input, test_image_output, test_residual_error = gen_img_set(batch = args['data_size'], grid_width= args['input_grid_width'], grid_height=args['input_grid_height'], num_time_steps=args['n_time_step'], show_sample_data = False)
        gat_train_dataset = GATDataset(state_input, action_input, image_input, image_output,residual_error)
        gat_test_dataset = GATDataset(test_state_input, test_action_input, test_image_input, test_image_output,test_residual_error)
        self.train_dataloader = DataLoader(gat_train_dataset, batch_size=args['batch_size'], shuffle=True)
        self.test_dataloader = DataLoader(gat_test_dataset, batch_size=args['batch_size'], shuffle=False)
        
    def load_ltatt_model(self,path=None):
        if self.ltatt_model_path is None and path is None:
            print("ltatt model load fail")
            return 
        
        if self.ltatt_model_path is not None:                
            ltatt_path = self.ltatt_model_path
        
        if path is not None:
            ltatt_path = path
            
        loaded_dict = torch.load(ltatt_path)            
        self.args = loaded_dict['args']
        self.auc_anp_rnn_model.load_ltatt_model(loaded_dict['model_state'])
        print("succefully load to ltatt model" + str(ltatt_path))
            
    def save_auc_anprnn_model(self,epoch = 0, path=None):
        if path is None:
            return
        if not os.path.exists(model_dir):
                    os.makedirs(model_dir)
                    
        checkpoint_path = os.path.join(model_dir, f'auc_anprnn_{epoch + 1}.pth')                            
        # Create a checkpoint dictionary including both model state and args
        checkpoint = {
            'model_state': self.auc_anp_rnn_model.state_dict(),
            'args': self.args
        }            
        torch.save(checkpoint, checkpoint_path)
        print(f"auc_anprnn model checkpoint saved at epoch {epoch + 1}")
        
    def train_ltatt_model(self):            
        device = self.device
        args = self.args            
        model = self.auc_anp_rnn_model.get_ltatt_model()
        train_dataloader = self.train_dataloader
        test_dataloader  = self.test_dataloader
        optimizer = Adam(model.parameters(), lr=0.001)        
        scheduler = MultiStepLR(optimizer, milestones=[0.75 * args['num_epochs'], 0.85 * args['num_epochs']], gamma=0.1)
        criterion = nn.MSELoss()  # Use MSE loss for pixel-wise comparison
        for epoch in range(args['num_epochs']):
            model.train()
            total_loss = 0.0
            for batch_idx, (state, action, image_in, image_out, residual_out) in enumerate(train_dataloader):
                state, action, image_in, image_out, residual_out= state.to(device).float(), action.to(device).float(), image_in.to(device).float(), image_out.to(device).float(), residual_out.to(device).float()
                optimizer.zero_grad()
                outputs = model(state, action,image_in) 
                loss = criterion(outputs, image_out)                    
                # Compute loss
                
                total_loss += loss.item() 

                # Backpropagation and optimization
                loss.backward()
                optimizer.step()

            avg_loss = total_loss / (batch_idx + 1)
            print(f"Epoch [{epoch + 1}/{args['num_epochs']}], Avg. Loss: {avg_loss:.6f}")

            # Log training loss to TensorBoard
            self.writer.add_scalar('LTATT Loss/Train', avg_loss, epoch + 1)
            
            # Save model checkpoint every 100 epochs
            if (epoch + 1) % 100 == 0:
                
                if not os.path.exists(model_dir):
                    os.makedirs(model_dir)
                checkpoint_path = os.path.join(model_dir, f'ltatt_{epoch + 1}.pth')                
                self.ltatt_model_path = checkpoint_path
                # Create a checkpoint dictionary including both model state and args
                checkpoint = {
                    'model_state': model.state_dict(),
                    'args': args
                }
                
                torch.save(checkpoint, checkpoint_path)
                print(f"ltatt model checkpoint saved at epoch {epoch + 1}")
                

            # Intermediate evaluation every 10 epochs
            if (epoch + 1) % 10 == 0:
                model.eval()
                test_loss = 0.0
                with torch.no_grad():
                    for batch_idx, (state, action, image_in, image_out, residual_out) in enumerate(test_dataloader):
                        state, action, image_in, image_out, residual_out = state.to(device).float(), action.to(device).float(), image_in.to(device).float(), image_out.to(device).float(), residual_out.to(device).float()
                        # Forward pass                                                        
                        outputs = model(state, action,image_in) 
                        loss = criterion(outputs, image_out)                                                                                
                        test_loss += loss.item() 

                avg_test_loss = test_loss / (batch_idx + 1)
                print(f"ltatt Test Avg. Loss: {avg_test_loss:.6f}")                    
                # Log test loss to TensorBoard
                self.writer.add_scalar('ltatt Loss/Test', avg_test_loss, epoch + 1)                                                        
            scheduler.step()
        
        
    def train_auc_anp_rnn_model(self):            
        device = self.device
        args = self.args            
        model = self.auc_anp_rnn_model
        train_dataloader = self.train_dataloader
        test_dataloader  = self.test_dataloader        
        optimizer = Adam(model.parameters(), lr=0.001)
        scheduler = MultiStepLR(optimizer, milestones=[0.75 * args['num_epochs'], 0.85 * args['num_epochs']], gamma=0.1)

        for epoch in range(args['num_epochs']):
            model.train()
            total_loss = 0.0
            ## image_out is the attention maksed applied image 
            for batch_idx, (state, action, image_in, image_out, residual_out) in enumerate(train_dataloader):
                state, action, image_in, image_out, residual_out= state.to(device).float(), action.to(device).float(), image_in.to(device).float(), image_out.to(device).float(), residual_out.to(device).float()
                # 50% to 80 % batch for target data
                context_data, target_data = get_context_target_data(state, action, image_in, residual_out)
                optimizer.zero_grad()
                mu, sigma, log_p, kl, loss = model(target_data, context_data) 
                total_loss += loss.item() 
                # Backpropagation and optimization
                loss.backward()
                optimizer.step()

            avg_loss = total_loss / (batch_idx + 1)
            print(f"Epoch [{epoch + 1}/{args['num_epochs']}], Avg. auc_anp_rnn Loss: {avg_loss:.6f}")

            # Log training loss to TensorBoard
            self.writer.add_scalar('AUC_ANP_RNN Loss/Train', avg_loss, epoch + 1)
            
            # Save model checkpoint every 100 epochs
            if (epoch + 1) % 100 == 0:
                checkpoint_path = os.path.join(model_dir, f'model_checkpoint_epoch_{epoch + 1}.pth')                
                self.save_auc_anprnn_model(epoch, path=checkpoint_path)
                
            # Intermediate evaluation every 10 epochs
            if (epoch + 1) % 10 == 0:
                model.eval()
                test_loss = 0.0
                with torch.no_grad():
                    for batch_idx, (state, action, image_in, image_out, residual_out) in enumerate(test_dataloader):
                        state, action, image_in, image_out, residual_out = state.to(device).float(), action.to(device).float(), image_in.to(device).float(), image_out.to(device).float(), residual_out.to(device).float()
                        context_data, target_data = get_context_target_data(state, action, image_in, residual_out)
                        # Forward pass
                        mu, sigma, log_p, kl, loss = model(target_data, context_data)                     
                        # Stop measuring forward pass time
                        # Compute loss
                        
                        test_loss += loss.item() 

                avg_test_loss = test_loss / (batch_idx + 1)
                print(f"auc_anp_rnn Test Avg. Loss: {avg_test_loss:.6f}")                
                # Log test loss to TensorBoard
                self.writer.add_scalar('auc_anp_rnn Loss/Test', avg_test_loss, epoch + 1)
                
            

            scheduler.step()
        # Close the SummaryWriter
        self.writer.close()


if __name__ == "__main__":
        # Initialize the AUCTrainAgent
    auc_train_agent = AUCTrainAgent()
    # Generate fake dataloaders
    auc_train_agent.gen_fake_dataloaders()
    
    
    model_load = False
    # Train the ltatt model
    auc_train_agent.train_ltatt_model()
    if model_load:
        ltatt_model_path = "path/to/ltatt_model.pth"
        auc_train_agent.load_ltatt_model(path=ltatt_model_path)

    # Train the AUC_ANP_RNN model
    auc_train_agent.train_auc_anp_rnn_model()
    