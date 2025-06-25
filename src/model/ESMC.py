import os
import esm
import torch
import torch.nn as nn

from esm.models.esmc import ESMC
from esm.tokenization import get_esmc_model_tokenizers



class Load_from_pretrained:
    def __init__(self, model_checkpoint):
        # load and define the model
        if 'esmc_300m' in model_checkpoint:
            checkpoint_path = '/stor/work/Wilke/wilkelab/pLMs_checkpoints/ESMC/esmc_300m_2024_12_v0.pth'
            self.model = self.ESMC_300M_202412(checkpoint_path)

        elif 'esmc_600m' in model_checkpoint:
            checkpoint_path = '/stor/work/Wilke/wilkelab/pLMs_checkpoints/ESMC/esmc_600m_2024_12_v0.pth'
            self.model = self.ESMC_600M_202412(checkpoint_path)
        
        else:
            raise ValueError("Invalid model name. Please use esmc_300m or esmc_600m.")
        

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

   
    
    def get_model_details(self):
        return self.model



if __name__ == "__main__":
    from argparse import ArgumentParser
    from src.data.dataset import MyDataModule

    args = ArgumentParser()
    args.add_argument('--checkpoint_path', type=str, default='/stor/work/Wilke/wilkelab/pLMs_checkpoints/ESMC/esmc_300m_2024_12_v0.pth')
    args.add_argument('-i', '--dataDir', type=str, default='data/processed/example/')
    args.add_argument('--batch_size', type=int, default=2)
    args = args.parse_args()
    
    print("Loading model for testing")
    data_module = MyDataModule(args)
    data_module.setup()
    train_loader = data_module.train_dataloader()

    model_name = 'ESM C' + args.checkpoint_path.split("/")[-1].split("_")[1]
    print(f"Model name: {model_name}")
    print(f"Model loaded from path: {args.checkpoint_path}\n")

    model = Load_from_pretrained(args.checkpoint_path)
    model= model.get_model_details()
    print(model)
      