import os
import esm
import torch
import torch.nn as nn



class Load_from_pretrained:
    def __init__(self, checkpoint_path):
        self.checkpoint_path = checkpoint_path
        self.model, self.alphabet = None, None
        self._load_model()

    def _load_model(self):
        """Load one one the pre-trained ESM-2 models, add the classification head"""
        supported_models = [
            'esm2_t6_8M_UR50D', 'esm2_t12_35M_UR50D', 'esm2_t30_150M_UR50D', 
            'esm2_t33_650M_UR50D', 'esm2_t36_3B_UR50D', 'esm2_t48_15B_UR50D'
            ]

        model_name = self.checkpoint_path.split('/')[-1].split('.')[0]
        # load from full path
        if self.checkpoint_path.endswith('.pt'):
            if not os.path.exists(self.checkpoint_path):
                raise FileNotFoundError(f"Checkpoint file not found: {self.checkpoint_path}")
            else:
                print(f"Loading model {model_name} from full path and preparing for fine-tuning...")
                self.model, self.alphabet = esm.pretrained.load_model_and_alphabet(self.checkpoint_path)
        # load from cache        
        elif self.checkpoint_path in supported_models:
            print(f"Loading model {self.checkpoint_path} and preparing for fine-tuning...")
            if self.checkpoint_path == 'esm2_t6_8M_UR50D':
                self.model, self.alphabet = esm.pretrained.esm2_t6_8M_UR50D()
            elif self.checkpoint_path == 'esm2_t12_35M_UR50D':
                self.model, self.alphabet = esm.pretrained.esm2_t12_35M_UR50D()
            elif self.checkpoint_path == 'esm2_t30_150M_UR50D':
                self.model, self.alphabet = esm.pretrained.esm2_t30_150M_UR50D()
            elif self.checkpoint_path == 'esm2_t33_650M_UR50D':
                self.model, self.alphabet = esm.pretrained.esm2_t33_650M_UR50D()
            elif self.checkpoint_path == 'esm2_t36_3B_UR50D':
                self.model, self.alphabet = esm.pretrained.esm2_t36_3B_UR50D()
            elif self.checkpoint_path == 'esm2_t48_15B_UR50D':
                print('This model is too big to be downloaded into the home directory.')
                print('Download it into the working repository and pass the full checkpoints path.')
        elif self.checkpoint_path not in supported_models:
            raise ValueError(f"Model {model_name} not supported. Supported models are: {supported_models}")
        
    
    def get_model_details(self):
        del self.model.contact_head
        return self.model, self.alphabet


if __name__ == "__main__":
    from argparse import ArgumentParser

    args = ArgumentParser()
    args.add_argument('--checkpoint_path', type=str, default='esm2_t33_650M_UR50D')
    args = args.parse_args()
    
    print("Loading model for testing")
    model_loader = Load_from_pretrained(args.checkpoint_path)
    model, alphabet = model_loader.get_model_details()
    
    print(f"Model loaded: {model}")
    print(f"Alphabet loaded: {alphabet}")