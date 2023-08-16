import torch
import torch.nn as nn
import numpy as np

from modules.AUC_lstm_model import AUCLSTMModel

from modules.latent_encoder import LatentEncoder
from modules.deterministic_encoder import DeterministicEncoder
from modules.decoder import Decoder 

# Define the model architecture with LSTM-based spatial attention
class AUCANPRNN_Model(nn.Module):    
    def __init__(self, args):
        super(AUCANPRNN_Model, self).__init__()
        
        load_AUC_lstm_model = False
        self.rnn = AUCLSTMModel(args)
       
        ## Params for AUC model 
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
        
        ## Params for AUC_ANP_RNN Model
        
        self.x_dim = self.auc_lstm_hidden_size
        self.y_dim = self.output_residual_dim        
        self.latent_dim = self.auc_lstm_hidden_size        
        self.use_deter_path = True
       
        self.mlp_hidden_size_list = [self.auc_lstm_hidden_size, 256, self.auc_lstm_hidden_size]
        
        use_self_attention = True
        le_self_attention_type="dot"
        de_self_attention_type="dot"
        de_cross_attention_type="multihead"
        use_deter_path=True
        
        # NOTICE: Latent Encoder
        self._latent_encoder = LatentEncoder(input_x_dim=self.x_dim,
                                             input_y_dim=self.y_dim,
                                             hidden_dim_list=self.mlp_hidden_size_list,
                                             latent_dim=self.latent_dim,
                                             use_self_attn=use_self_attention,
                                             self_attention_type=le_self_attention_type
                                             )
        # NOTICE : Decoder
        self._decoder = Decoder(x_dim=self.x_dim,
                                y_dim=self.y_dim,
                                mid_hidden_dim_list=self.mlp_hidden_size_list,
                                latent_dim=self.latent_dim,  # the dim of last axis of sc and z..
                                use_deterministic_path=True,  # whether use d_path or not will change the size of input
                                use_lstm=False
                                 )

        # NOTICE: Deterministic Encoder
        self._deter_encoder = DeterministicEncoder( input_x_dim=self.x_dim,
                                                    input_y_dim=self.y_dim,
                                                    hidden_dim_list=self.mlp_hidden_size_list,
                                                    latent_dim=self.latent_dim,  # the dim of last axis of r..
                                                    self_attention_type=de_self_attention_type,
                                                    use_self_attn=use_self_attention,
                                                    attention_layers=2,
                                                    use_lstm=False,
                                                    cross_attention_type=de_cross_attention_type,
                                                    attention_dropout=0)

        
       
        self.context_x = None
        self.context_y = None
    
    def get_ltatt_model(self):
        return self.rnn.local_temp_attention_module
    
    def load_ltatt_model(self,state_dict):
        self.rnn.load_ltatt_model(state_dict)
        
    def set_context_x_y(self,state, action_predictions,image , residual_ouput):
        context_x, context_y = self.data_to_context(state, action_predictions,image)
        self.context_x = context_x.clone()
        self.context_y = context_y.clone()
    
    def data_to_context(self,state, action_predictions,image , residual_ouput):
        _, context_x = self.rnn(state, action_predictions,image)
        context_x = context_x
        context_y = residual_ouput
        return context_x.clone(), context_y.clone()
    
    def _get_conv_out_size(self, model, width,height, input_channels, out_channels, kernel_size, stride):
        dummy_input = torch.randn(1, input_channels, width,height).to(self.device).to(torch.float)   # Assuming input size of (64, 64)
        model = model.to(self.device).to(torch.float)
        conv_output = model(dummy_input)
        return conv_output.view(-1).size(0)
    
    #       
    
    def augment_target(self,target_x,target_y,context_x, context_y):        
        original_target_x_batch_size = target_x.shape[0]        
        new_shape = context_x.shape
        new_tensor = torch.zeros(*new_shape).to(device=self.device).to(torch.float)        
        new_tensor[:target_x.shape[0], :, :] = target_x
        new_tensor[target_x.shape[0]:, :, :] = target_x[-1,:,:]
        
        new_shape_y = context_y.shape
        new_tensor_y = torch.zeros(*new_shape_y).to(device=self.device).to(torch.float)        
        new_tensor_y[:target_x.shape[0], :, :] = target_y
        new_tensor_y[target_x.shape[0]:, :, :] = target_y[-1,:,:]
            
        return new_tensor, new_tensor_y
        
        
    def forward(self, target_data, context_data = None):
        
        if context_data is not None:
            state, action_predictions, image, residual_ouput = context_data                        
            context_x, context_y = self.data_to_context(state, action_predictions, image, residual_ouput)
        else:
            context_x = self.context_x 
            context_y = self.context_y
        
        target_state, target_action_predictions, target_image, target_residual_ouput = target_data
        
        target_x, target_y = self.data_to_context(target_state, target_action_predictions, target_image, target_residual_ouput)
        
        if target_x.shape[0] < context_x.shape[0]:
            target_x, target_y = self.augment_target(target_x,target_y,context_x, context_y) 
            
            
            
            
        _, target_size, _ = target_x.size()
        
        ##################### ANP RNN INIT
        
        prior_dist, prior_mu, prior_sigma = self._latent_encoder(context_x, context_y)

        # For training, when target_y is available, use targets for latent encoder.
        # Note that targets contain contexts by design.
        # NOTICE: Here is the difference:
        #   Deepmind: latent_rep = prior/poster .sample()
        #   soobinseo: latent_rep = prior/poster
        #   3spring :  latent_rep = prior/poster .loc
        # TODO: loc will work, change to sample later
        if target_y is not None:
            # NOTICE: Training      *(context = test) for neural process
            post_dist, post_mu, post_sigma = self._latent_encoder(target_x, target_y)
            Z = post_dist.loc
        else:
            # NOTICE: Testing
            Z = prior_dist.loc
        # Z (b, latent_dim)

        # print('before unsequeeze, Z.size() =', Z.size())

        Z = Z.unsqueeze(1).repeat(1, target_size, 1)
        # Z (b, target_size, latent_dim) verified

        # print('after unsequeeze, Z.size() =', Z.size())

        # NOTICE: obtain r using deterministic path                
        ## target x has to have the same batch as the context
        
        if self.use_deter_path:
            R = self._deter_encoder(context_x, context_y, target_x)
            # R (B, target_size, latent_dim)
        else:
            R = None

        # Obtain the prediction
        dist, mu, sigma = self._decoder(R, Z, target_x)

        # If we want to calculate the log_prob for training we will make use of the
        # target_y. At test time the target_y is not available so we return None.
        if target_y is not None:
            # get log probability
            # Get KL between prior and posterior
            kl = torch.distributions.kl_divergence(post_dist, prior_dist).mean(-1)

            log_p = dist.log_prob(target_y).mean(-1)
            # print('log_p.size() =', log_p.size())
            # log_p = dist.log_prob(target_y).mean(-1)
            loss_kl = kl[:, None].expand(log_p.shape)
            # print('torch.mean(loss_kl) =', torch.mean(loss_kl))
            loss = - (log_p - loss_kl).mean()
        else:
            log_p = None
            kl = None
            loss = None

        return mu, sigma, log_p, kl, loss

