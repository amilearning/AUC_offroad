import torch
from torch.utils.data import Dataset

class GATDataset(Dataset):    
    def __init__(self, state_input, action_input,img_input, img_output, residual_output):
        
        
        
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.img_input = torch.tensor(img_input).to(self.device).float()
        # self.state_input = torch.tensor(state_input).to(self.device).float()
        # self.action_input = torch.tensor(action_input).to(self.device).float()
        # self.img_output = torch.tensor(img_output).to(self.device).float()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.img_input = img_input
        self.state_input = state_input
        self.action_input = action_input
        self.img_output = img_output
        self.residual_output = residual_output
        

        assert len(self.img_input) == len(self.state_input) == len(self.action_input) == len(self.img_output) == len(self.residual_output), "All input and output must have the same length"

    # def collate_fn(self, batch):
    #     # Custom collate function to convert data to tensors and move to CUDA

    #     # Assuming your dataset returns tuples (image, label)
    #     state_input, action_input,img_input,  img_output = zip(*batch)
    #     state_input = state_input.to(self.device).float()
    #     action_input = action_input.to(self.device).float()        
    #     img_input = img_input.to(self.device).float()
    #     img_output = img_output
    #     # # Convert to tensors, move to CUDA, and cast to float
    #     # images = torch.stack(images).to(self.device).float()
    #     # labels = torch.tensor(labels).to(self.device).float()

    #     return state_input, action_input,img_input,  img_output



    def __len__(self):
        return len(self.img_input)

    def __getitem__(self, idx):
        img_input = self.img_input[idx]
        state_input = self.state_input[idx]
        action_input = self.action_input[idx]
        img_output = self.img_output[idx]
        residual_output = self.residual_output[idx]

        return state_input, action_input,img_input,  img_output, residual_output