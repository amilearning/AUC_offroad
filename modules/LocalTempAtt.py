import torch
import torch.nn as nn
import numpy as np

from sklearn.model_selection import train_test_split

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # avg_out = torch.mean(x, dim=1, keepdim=True)
        # max_out, _ = torch.max(x, dim=1, keepdim=True)
        # x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)
    
    
''' 
This module extracts the predicted patch given the current state and the predicted inputs 
Input: state, action, image 
Output: maksed image for predicted time step 
'''
# Define the model architecture with LSTM-based spatial attention
class LocalTemporalAttention(nn.Module):
    def __init__(self, args):
        super(LocalTemporalAttention, self).__init__()
        self.input_grid_width = args['input_grid_width']
        self.input_grid_height = args['input_grid_height']
        self.input_state_dim = args['input_state_dim']
        self.input_action_dim = args['input_action_dim']
        self.n_time_step = args['n_time_step']
        self.init_fc_hidden_size = args['init_fc_hidden_size']
        self.lstm_hidden_size = args['lstm_hidden_size']
        self.output_residual_dim = args['output_residual_dim']
        self.lstm_input_size = self.lstm_hidden_size + self.input_action_dim
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.init_input_size = self.input_grid_width*self.input_grid_height + self.input_state_dim + self.input_action_dim        
        
        
        # Convolutional layer        
        conv_kernel_size = 3
        conv_stride = 1
        conv_int_ch = 1 # test with 1d image, but will be at least 4 (RGB-D)
        conv_out_ch = 1
        self.init_conv_layer = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=conv_kernel_size, stride=conv_stride),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # Optional pooling layer
        ).to(device= self.device)
        
        # Calculate the number of features after convolution
        self.conv_out_size = self._get_conv_out_size(self.input_grid_width, self.input_grid_height,1 , 1, conv_kernel_size, conv_stride)
        
        self.init_conv_layer_fc = nn.Sequential(              
                    nn.Linear(self.conv_out_size, self.init_input_size-self.input_state_dim - self.input_action_dim ),
                    nn.ReLU()
        )
                
        self.intput_to_lstm_fc = nn.Sequential(              
                nn.Linear(self.init_input_size, self.init_fc_hidden_size),  
                nn.BatchNorm1d(self.init_fc_hidden_size),
                nn.ReLU(),
                nn.Linear(self.init_fc_hidden_size, self.lstm_input_size),  
                nn.BatchNorm1d(self.lstm_input_size),
                nn.ReLU()                
        )    
        
        # Define the LSTM layer  
        self.lstm = nn.LSTM(input_size=self.lstm_input_size,  
                    hidden_size=self.lstm_hidden_size,
                    num_layers=1,
                    batch_first=True)

        hidden_lstm_channel = 1        
        lstm_hidden_fc_hidden = 20
        self.lstm_hidden_fc = nn.Sequential(
                nn.Linear(self.lstm_hidden_size, lstm_hidden_fc_hidden),  
                nn.BatchNorm1d(lstm_hidden_fc_hidden),
                nn.ReLU(),
                nn.Linear(lstm_hidden_fc_hidden, hidden_lstm_channel *self.input_grid_width*self.input_grid_height),  
                nn.BatchNorm1d(hidden_lstm_channel *self.input_grid_width*self.input_grid_height),
                nn.ReLU()                
        )        
                # Attention mechanism
        self.grid_attention = SpatialAttention()
    
    def _get_conv_out_size(self, width,height, input_channels, out_channels, kernel_size, stride):
        dummy_input = torch.randn(1, input_channels, width,height).to(self.device).to(torch.float)   # Assuming input size of (64, 64)
        conv_output = self.init_conv_layer(dummy_input)
        return conv_output.view(-1).size(0)
        
    def forward(self, state, action_predictions,image ):
        # Initialize the LSTM hidden state
        
        outputs = []
        batch_size = state.shape[0]
        
        # Concatenate state and action predictions along the sequence dimension
        init_lstm_input = torch.cat((state, action_predictions[:,0,:]), dim=1).to(self.device).to(torch.float) 
        # flattened_batch_image_data = image.view(state.shape[0], -1)
        tmp = self.init_conv_layer(image.unsqueeze(dim=1))
        tmp = tmp.view(batch_size, -1)
        flattened_batch_image_data = self.init_conv_layer_fc(tmp)
        init_x = torch.cat([init_lstm_input,flattened_batch_image_data], dim=1).to(self.device).to(torch.float)
        
        lstm_x = self.intput_to_lstm_fc(init_x)
        

        init_input_to_lstm = lstm_x.unsqueeze(dim=1)
        
        h0 = torch.randn(self.lstm.num_layers, init_input_to_lstm.shape[0], self.lstm_hidden_size).to(self.device).to(torch.float)  # hidden states
        c0 = torch.randn(self.lstm.num_layers, init_input_to_lstm.shape[0], self.lstm_hidden_size).to(self.device).to(torch.float)  # cell states
        init_lstm_output, (h, c) = self.lstm(init_input_to_lstm, (h0, c0))
        
           
        init_flatten_hidden = self.lstm_hidden_fc(h[-1,:,:])
        # Reshape for convolutional layer (batch_size, channels, height, width)
        fc_output = init_flatten_hidden.view(-1, self.input_grid_width, self.input_grid_height)
        # add additional dimension to make 1 channel data
        fc_output =fc_output.unsqueeze(dim=1)
        ## apply attention 
        attentioned_fc_output = self.grid_attention(fc_output)*image.unsqueeze(dim=1)

        outputs.append(attentioned_fc_output.squeeze().clone())
        
        for t in range(1,action_predictions.shape[1]):
            if t ==1:
                next_input_to_lstm = torch.cat((init_lstm_output,action_predictions[:, t:t+1]),dim=2).to(self.device).to(torch.float)            
            else:
                next_input_to_lstm = torch.cat((next_input_to_lstm,action_predictions[:, t:t+1]),dim=2).to(self.device).to(torch.float)            
            next_input_to_lstm, (h, c) = self.lstm(next_input_to_lstm,(h,c))
            
            
            init_flatten_hidden = self.lstm_hidden_fc(h[-1,:,:])
            # Reshape for convolutional layer (batch_size, channels, height, width)
            fc_output = init_flatten_hidden.view(-1, self.input_grid_width, self.input_grid_height)
            # add additional dimension to make 1 channel data
            fc_output =fc_output.unsqueeze(dim=1)
            ## apply attention 
            attentioned_fc_output = self.grid_attention(fc_output)*image.unsqueeze(dim=1)
            outputs.append(attentioned_fc_output.squeeze().clone())
        

        outputs = torch.stack(outputs,dim = -1)

        return outputs


    def load_model(self,path):
        checkpoint = torch.load(path)
        new_model.load_state_dict(loaded_dict['model_state'])
        
        