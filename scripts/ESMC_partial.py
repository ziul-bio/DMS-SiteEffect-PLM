import os
import esm
import torch
import torch.nn as nn

from esm.models.esmc import ESMC
from esm.tokenization import get_esmc_model_tokenizers



class ClassificationHead(nn.Module):
    def __init__(self, embed_dim, num_classes, hidden_dropout):
        super(ClassificationHead, self).__init__()
        self.dense = nn.Linear(embed_dim, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)  # Normalize hidden states
        self.out_proj = nn.Linear(embed_dim, num_classes) 
        self.dropout = nn.Dropout(hidden_dropout)
        self.gelu = nn.GELU()  # Use GELU activation function

    def forward(self, features):
        x = features[:, 0, :]  # CLS token
        x = self.dense(x)
        x = self.layer_norm(x)  # Helps stabilize training
        x = self.gelu(x)        # this should be optional, based on the model. For my regression tasks it did not worked well
        x = self.dropout(x) 
        logits = self.out_proj(x)
        return logits



class RegressionHead(nn.Module):
    def __init__(self, embed_dim, num_classes, hidden_dropout):
        super(RegressionHead, self).__init__()
        self.dense = nn.Linear(embed_dim, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)  # Normalize hidden states
        self.out_proj = nn.Linear(embed_dim, num_classes) 
        self.dropout = nn.Dropout(hidden_dropout)
        self.gelu = nn.GELU()  # Use GELU activation function

    def forward(self, features):
        x = features[:, 0, :]  # CLS token
        x = self.dense(x)
        x = self.layer_norm(x)  # Helps stabilize training
        x = self.gelu(x)  
        x = self.dropout(x) 
        logits = self.out_proj(x)
        return logits


class Load_from_pretrained:
    def __init__(self, checkpoint_path, num_classes, dropout):
        self.checkpoint_path = checkpoint_path
        self.num_classes = num_classes
        self.dropout = dropout

        # load and define the model
        if 'esmc_300m' in self.checkpoint_path:
            self.model = self.ESMC_300M_202412(self.checkpoint_path)
            self.model_dimension = 960
            self.tokenizer = self.model._tokenize

        elif 'esmc_600m' in self.checkpoint_path:
            self.model = self.ESMC_600M_202412(self.checkpoint_path)
            self.model_dimension = 1152
            self.tokenizer = self.model._tokenize
        
        elif 'vicam_300m' in self.checkpoint_path:
            self.model = self.VICAM_300M(self.checkpoint_path)
            self.model_dimension = 960
            self.tokenizer = self.model._tokenize

        elif 'vicam_600m' in self.checkpoint_path:
            self.model = self.VICAM_600M(self.checkpoint_path)
            self.model_dimension = 1152
            self.tokenizer = self.model._tokenize
        else:
            raise ValueError("Invalid model name. Please use esmc_300m_2024_12_v0.pth or esmc_600m_2024_12_v0.pth.")

        self.setup_model_for_tune()



    # load the models locally
    def ESMC_300M_202412(self, model_path: str, device: torch.device | str = "cpu"):
        device = torch.device(device)
        model = ESMC(
            d_model=960, n_heads=15, n_layers=30, tokenizer=get_esmc_model_tokenizers()
            )
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        # Convert model parameters to torch.bfloat16 or torch.float32
        model = model.to(torch.float32)
        return model

    def ESMC_600M_202412(self, model_path: str, device: torch.device | str = "cpu"):
        device = torch.device(device)
        model = ESMC(
            d_model=1152, n_heads=18, n_layers=36, tokenizer=get_esmc_model_tokenizers()
        )
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        # Convert model parameters to float32
        model = model.to(torch.float32)
        return model

    def VICAM_300M(self, model_path: str, device: torch.device | str = "cpu"):
        device = torch.device(device)
        model = ESMC(
            d_model=960, n_heads=15, n_layers=30, tokenizer=get_esmc_model_tokenizers()
            )
        state_dict = torch.load(model_path, map_location=device, weights_only=True)["state_dict"]
        new_state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict)
        # Convert model parameters to torch.bfloat16 or torch.float32
        model = model.to(torch.float32)
        return model

    def VICAM_600M(self, model_path: str, device: torch.device | str = "cpu"):
        device = torch.device(device)
        model = ESMC(
            d_model=1152, n_heads=18, n_layers=36, tokenizer=get_esmc_model_tokenizers()
            )
        state_dict = torch.load(model_path, map_location=device, weights_only=True)["state_dict"]
        new_state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict)
        # Convert model parameters to torch.bfloat16 or torch.float32
        model = model.to(torch.float32)
        return model

   
    
    def setup_model_for_tune(self):
        # freeze all layers but last one
        print("Freezing all layers but the last two...")
        num_layers = len(self.model.transformer.blocks)
        n_trainable = 2
        # trainable_blocks = ["transformer.blocks.28.", "transformer.blocks.29."]
        trainable_blocks = [f"transformer.blocks.{i}." for i in range(num_layers - n_trainable, num_layers)] # from last-2 to last layer
        for name, param in self.model.named_parameters():
            param.requires_grad = any(block in name for block in trainable_blocks)
        
        # print the trainable parameters
        #trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        #print(f"Number of trainable parameters: {trainable_params}")

        # Change the classification head to match the number of classes, regression or classification
        if self.num_classes == 1:
            print('Adding a new regression head...')
            self.model.sequence_head = RegressionHead(self.model_dimension, self.num_classes, self.dropout)
        else:
            print('Adding a new classification head...')
            self.model.sequence_head = ClassificationHead(self.model_dimension, self.num_classes, self.dropout)
    
    def get_model_details(self):
        return self.model, self.tokenizer



if __name__ == "__main__":
    from argparse import ArgumentParser
    from src.data.dataset import MyDataModule
    args = ArgumentParser()
    args.add_argument('--checkpoint_path', type=str, default='/stor/work/Wilke/wilkelab/pLMs_checkpoints/ESMC/esmc_300m_2024_12_v0.pth')
    #args.add_argument('--checkpoint_path', type=str, default='checkpoints/vicam_300m/CRVDBv29_maxLen2046_20aa_Full_RLRP_lr1e6/epoch=9-val_loss=1.52.ckpt')
    args = args.parse_args()


    model = Load_from_pretrained(args.checkpoint_path, num_classes=1, dropout=0.1)
    model= model.get_model_details()

    for name, param in model[0].named_parameters():
        if param.requires_grad:
            print(name, param.requires_grad)
    
    #print(model)