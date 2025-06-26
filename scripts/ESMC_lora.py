import os
import esm
import torch
import torch.nn as nn

from esm.models.esmc import ESMC
from esm.tokenization import get_esmc_model_tokenizers
from peft import get_peft_model, LoraConfig



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
        #self.gelu = nn.GELU()  # Use GELU activation function

    def forward(self, features):
        x = features[:, 0, :]  # CLS token
        x = self.dense(x)
        x = self.layer_norm(x)  # Helps stabilize training
        #x = self.gelu(x)  
        x = self.dropout(x) 
        logits = self.out_proj(x)
        return logits


class Load_from_pretrained:
    def __init__(self, checkpoint_path, num_classes, dropout, lora_r, lora_alpha, lora_dropout, lora_modules):
        self.checkpoint_path = checkpoint_path
        self.num_classes = num_classes
        self.dropout = dropout

        # LoRA parameters
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.lora_modules = lora_modules
        self.peft_config = LoraConfig(
            inference_mode=False,
            bias="none",
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=self.lora_modules)


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
        self.model = self._add_lora_layers()



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
        # Change the classification head to match the number of classes, regression or classification
        if self.num_classes == 1:
            print('Adding a new regression head...')
            self.model.sequence_head = RegressionHead(self.model_dimension, self.num_classes, self.dropout)
        else:
            print('Adding a new classification head...')
            self.model.sequence_head = ClassificationHead(self.model_dimension, self.num_classes, self.dropout)
    
    
    
    def _add_lora_layers(self):
        """Add LoRA layer to the model"""
        print(f"Lora configuration: {self.peft_config}")
        self.model = get_peft_model(self.model, self.peft_config)
        
        # unfreeze the lm_head layer for fine-tuning
        for param in self.model.base_model.model.sequence_head.parameters():
            param.requires_grad = True
        return self.model
    
    
    def get_model_details(self):
        return self.model, self.tokenizer



if __name__ == "__main__":
    from argparse import ArgumentParser
    from src.data.dataset import MyDataModule
    args = ArgumentParser()
    args.add_argument('--checkpoint_path', type=str, default='/stor/work/Wilke/wilkelab/pLMs_checkpoints/ESMC/esmc_300m_2024_12_v0.pth')
    #args.add_argument('--checkpoint_path', type=str, default='checkpoints/vicam_300m/CRVDBv29_maxLen2046_20aa_Full_RLRP_lr1e6/epoch=9-val_loss=1.52.ckpt')
    args.add_argument("--num_classes", type=int, default=1, help="Number of classes (1 for regression).")
    args.add_argument('--batch_size', type=int, default=2)
    
    args.add_argument('--lora_r', type=int, default=4, help='Rank of the low-rank decomposition.')
    args.add_argument('--lora_alpha', type=int, default=32, help='Scaling factor for LoRA weights. alpha/rank.')
    args.add_argument('--lora_dropout', type=float, default=0.01, help='Dropout rate for LoRA.')
    args.add_argument('--lora_modules', nargs='*', type=str, default=["attn.layernorm_qkv.1"])
    args = args.parse_args()
    
    
    model = Load_from_pretrained(args.checkpoint_path, num_classes=1, dropout=0.1, lora_r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout, lora_modules=args.lora_modules)
    model= model.get_model_details()
    print(model)

