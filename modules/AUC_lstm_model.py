import torch
import torch.nn as nn
import numpy as np
from modules.LocalTempAtt import LocalTemporalAttention

'''
This module will be used by AUC_anprnn_model 
Input: state, image, action prediciton 
Output: Hidden states for residual predicted positions 
'''
  
# Define the model architecture with LSTM-based spatial attention
class AUCLSTMModel(nn.Module):    
    def __init__(self, args):
        super(AUCLSTMModel, self).__init__()
        
        self.local_temp_attention_module = LocalTemporalAttention(args)
        
        
        self.input_grid_width = args['input_grid_width']
        self.input_grid_height = args['input_grid_height']
        self.input_state_dim = args['input_state_dim']
        self.input_action_dim = args['input_action_dim']
        self.n_time_step = args['n_time_step']
        self.init_fc_hidden_size = args['init_fc_hidden_size']
        self.lstm_hidden_size = args['lstm_hidden_size']
        self.output_residual_dim = args['output_residual_dim']
        self.lstm_input_size = self.lstm_hidden_size + self.input_action_dim
        self.auc_lstm_hidden_size = 20
        self.auc_lstm_input_dim = self.auc_lstm_hidden_size + self.input_action_dim
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.init_input_size = self.input_grid_width*self.input_grid_height + self.input_state_dim + self.input_action_dim        
        # #TODO ## assume we have 1 channel ############
        
        # # Convolutional layer  
        '''
        This layer takes the attended image set and convert it to 
        '''      
        conv_kernel_size = 3
        conv_stride = 1        
        self.att_image_to_lstm_conv_layer = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=conv_kernel_size, stride=conv_stride),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        ).to(self.device).to(torch.float)  # Optional pooling layer
        self.auc_conv_out_size = self._get_conv_out_size(self.att_image_to_lstm_conv_layer,self.input_grid_width, self.input_grid_height,1 , 1, conv_kernel_size, conv_stride)        
        
        self.init_att_image_to_lstm_fc_layer = nn.Sequential(            
            nn.Linear(self.auc_conv_out_size, self.auc_lstm_input_dim-self.input_action_dim-self.input_state_dim),
            nn.BatchNorm1d(self.auc_lstm_input_dim-self.input_action_dim-self.input_state_dim),
            nn.ReLU()
        ).to(self.device).to(torch.float)
                
        self.att_image_to_lstm_fc_layer = nn.Sequential(
            nn.Linear(self.auc_conv_out_size, self.auc_lstm_input_dim-self.input_action_dim),
            nn.BatchNorm1d(self.auc_lstm_input_dim-self.input_action_dim),
            nn.ReLU()
        ).to(self.device).to(torch.float)

        # Define the Second LSTM layer for estimating the residual position errors
        self.auc_lstm = nn.LSTM(input_size=self.auc_lstm_input_dim,  
                    hidden_size=self.auc_lstm_hidden_size,
                    num_layers=1,
                    batch_first=True).to(self.device).to(torch.float)
        
        
        self.auc_output_fc_hidden = 5        
        self.auc_output_fc = nn.Sequential(
                nn.Linear(self.auc_lstm_hidden_size, self.auc_output_fc_hidden),  
                nn.BatchNorm1d(self.auc_output_fc_hidden),
                nn.ReLU(),
                nn.Linear(self.auc_output_fc_hidden, self.output_residual_dim)                
        ).to(self.device).to(torch.float)   
        
        
        # # Calculate the number of features after convolution 
     
        

    def load_ltatt_model(self,state_dict):
        self.local_temp_attention_module.load_state_dict(state_dict)
    
    def _get_conv_out_size(self, model, width,height, input_channels, out_channels, kernel_size, stride):
        dummy_input = torch.randn(1, input_channels, width,height).to(self.device).to(torch.float)   # Assuming input size of (64, 64)
        model = model.to(self.device).to(torch.float)
        conv_output = model(dummy_input)
        return conv_output.view(-1).size(0)
    
        
    def forward(self, state, action_predictions,image ):
        pred_attended_images = []
        
        hidden_outputs = []
        batch_size = state.shape[0]
        
        '''
        ### TODO:Currently, LocalTemporalAttention and the proposed ModelErrorQuantifier
        ## runs sequentially but it can be run in parallel --> will improve the computation time
        '''
        # attended_images have [batch , image_width, image_height, sequence]
        attended_images = self.local_temp_attention_module(state, action_predictions,image)
        pred_attended_images = attended_images
        #############################################
        
        
        #####################################################################################
        #######################   init LSTM2 for residual error #############################
        #####################################################################################
        # compute the inital features for the second LSTM
        
        init_lstm_input_state_action = torch.cat((state, action_predictions[:,0,:]), dim=1).to(self.device).to(torch.float) 
        # init_lstm_input - > 
        
        init_conv_result = self.att_image_to_lstm_conv_layer(attended_images[:,:,:,0].unsqueeze(dim=1))
        init_conv_fc_result = self.init_att_image_to_lstm_fc_layer(init_conv_result.view(batch_size,-1))
        init_lstm_input = torch.cat([init_lstm_input_state_action,init_conv_fc_result], dim=1).to(self.device).to(torch.float)
        
        # Initialize the LSTM hidden state
        h0 = torch.randn(self.auc_lstm.num_layers, batch_size, self.auc_lstm_hidden_size).to(self.device).to(torch.float)  # hidden states
        c0 = torch.randn(self.auc_lstm.num_layers, batch_size, self.auc_lstm_hidden_size).to(self.device).to(torch.float)  # cell states        
        
        # initial guess from LSTM
        init_lstm_output, (h,c) = self.auc_lstm(init_lstm_input.unsqueeze(dim=1),(h0,c0))
        
        hidden_outputs.append(h[-1,:,:].clone())
        
        #####################################################################################
        #######################    LSTM for getting hidden states error   #######################
        #####################################################################################
        
        for t in range(1,action_predictions.shape[1]):
          
            conv_result = self.att_image_to_lstm_conv_layer(attended_images[:,:,:,t].unsqueeze(dim=1))
            conv_fc_result = self.att_image_to_lstm_fc_layer(conv_result.view(batch_size,-1))
            init_lstm_input_state_action = torch.cat((state, action_predictions[:,0,:]), dim=1).to(self.device).to(torch.float) 
            lstm_input = torch.cat([action_predictions[:,t,:],conv_fc_result], dim=1).to(self.device).to(torch.float) 
            lstm_output, (h,c) = self.auc_lstm(lstm_input.unsqueeze(dim=1),(h,c))        
            hidden_outputs.append(h[-1,:,:].clone())
        
            
        #####################################################################################
        #######################   LSTM for getting hidden states  END  #######################
        #####################################################################################

        hidden_outputs = torch.stack(hidden_outputs,dim = 1)
        pred_attended_imagess = pred_attended_images.permute(0, 3, 1, 2)
        # outputs are having [batch, sequence, features, (optional, feature for image)]
        return pred_attended_imagess, hidden_outputs

