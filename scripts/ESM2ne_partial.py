import os
import esm
import torch
import torch.nn as nn

import argparse
import pandas as pd
import pytorch_lightning as pl
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader


def parse_args():
    parser = argparse.ArgumentParser(description="Training script arguments.")
    parser.add_argument("-i", "--data", type=str, required=True, help="Path to input CSV file.")
    parser.add_argument("--checkpoint", type=str, default="esm2_t33_650M_UR50D", help="Model checkpoint name or full path.")
    parser.add_argument("--num_classes", type=int, default=1, help="Number of classes (1 for regression).")
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"), help="Device for training.")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size.")
    parser.add_argument("--dropout", type=float, default=0.1, help="CLS dropout probability.")
    return parser.parse_args()



# class MyDataset(Dataset):
#     """This class just loads the data and return a dataset object, returning the sequences and targets.
#     Without any tokenization or padding.
#     This will be handled later in the collate_fn.
#     """
#     def __init__(self, data_file):
#         self.scaler = StandardScaler()
#         self.data = data_file
#         self.sequences = [(d['ID'], d['sequence']) for _, d in self.data.iterrows()]
#         self.targets = self.scaler.fit_transform(self.data['target'].to_frame()).squeeze()
     
#     def __len__(self):
#         return len(self.targets)

#     def __getitem__(self, idx):
#         sequences = self.sequences[idx]
#         target = torch.tensor(self.targets[idx], dtype=torch.float).unsqueeze(-1)
        
#         return sequences, target



# class MyDataModule(pl.LightningDataModule):
#     def __init__(self, tokenizer, args):
#         super().__init__()
#         self.args = args
#         self.alphabet = tokenizer
        
#     def setup(self, stage=None):
#         data = pd.read_csv(args.data)

#         ## Use random_split to split the dataset into train and validation sets
#         dataset = MyDataset(data)
#         train_size = int(0.8 * len(dataset))
#         val_size = len(dataset) - train_size
#         self.train_dataset, self.val_dataset = random_split(
#             dataset, [train_size, val_size],
#             generator=torch.Generator().manual_seed(42)
#             )
       
#     def collate_fn(self, batch):
#         """This function will be used to collate the data into a batch.
#         It will handle the tokenization and padding of the sequences.
#         """
#         # batch: a list of ( (ID, sequence), target )
#         seqs = [item[0] for item in batch]
#         targets = [item[1] for item in batch]
    
#         # Tokenize the sequences
#         batch_converter = self.alphabet.get_batch_converter()
#         batch_labels, batch_strs, batch_tokens = batch_converter(seqs)
#         #batch_lens = (batch_tokens != self.alphabet.padding_idx).sum(1)
#         targets = torch.stack(targets) # Stack the targets to a shape (batch_size, 1)
#         return batch_tokens, targets

#     def train_dataloader(self):
#         return DataLoader(
#             self.train_dataset,
#             batch_size=self.args.batch_size,
#             shuffle=True,
#             collate_fn=self.collate_fn,
#             num_workers=4,
#             pin_memory=True,
#             drop_last=True,
#         )

#     def val_dataloader(self):
#         return DataLoader(
#             self.val_dataset,
#             batch_size=self.args.batch_size,
#             collate_fn=self.collate_fn,
#             num_workers=4,
#             pin_memory=True,
#             drop_last=True,
#         )





class RegressionHead(nn.Module):
    def __init__(self, embed_dim, num_classes, hidden_dropout):
        super(RegressionHead, self).__init__()
        self.dense = nn.Linear(embed_dim, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)  
        self.out_proj = nn.Linear(embed_dim, num_classes) 
        self.dropout = nn.Dropout(hidden_dropout)
        self.gelu = nn.GELU()  

    def forward(self, representation):
        #x = representation[:, 0, :]  # CLS token
        x = representation[:, 1:-1, :].mean(dim=1)  # mean representation, skiping CLS (0) and EOS (-1). Only possible because in these DMS all seqs have the same length.
        x = self.dense(x)
        x = self.layer_norm(x)
        x = self.gelu(x)  
        x = self.dropout(x) 
        logits = self.out_proj(x)
        return logits



class LoadFromPretrained:
    def __init__(self, checkpoint, num_classes, hidden_dropout):
        self.checkpoint = checkpoint
        self.num_classes = num_classes
        self.hidden_dropout = hidden_dropout
        self.model, self.alphabet = None, None
        self.model_dimension = 0
        self._load_model()
        self.setup_model_for_tune()

    def _load_model(self):
        """Load one one the pre-trained ESM-2 models, add the classification head"""
        
        supported_models = [
            'esm2_t6_8M_UR50D', 'esm2_t12_35M_UR50D', 'esm2_t30_150M_UR50D', 
            'esm2_t33_650M_UR50D', 'esm2_t36_3B_UR50D', 
            'esm2_viral_650M',
            ]
        
        # load from cache        
        if self.checkpoint in supported_models:
            print(f"Loading model {self.checkpoint} and preparing for fine-tuning...")
            if self.checkpoint == 'esm2_t6_8M_UR50D':
                self.model, self.alphabet = esm.pretrained.esm2_t6_8M_UR50D()
            elif self.checkpoint == 'esm2_t12_35M_UR50D':
                self.model, self.alphabet = esm.pretrained.esm2_t12_35M_UR50D()
            elif self.checkpoint == 'esm2_t30_150M_UR50D':
                self.model, self.alphabet = esm.pretrained.esm2_t30_150M_UR50D()
            elif self.checkpoint == 'esm2_t33_650M_UR50D':
                self.model, self.alphabet = esm.pretrained.esm2_t33_650M_UR50D()
            elif self.checkpoint == 'esm2_t36_3B_UR50D':
                self.model, self.alphabet = esm.pretrained.esm2_t36_3B_UR50D()
            
            # fine tuned models
            elif self.checkpoint == 'esm2_viral_650M':
                self.model, self.alphabet = esm.pretrained.esm2_t33_650M_UR50D()
                del self.model.contact_head
                model_checkpoint = '/stor/work/Wilke/luiz/ViCAM/checkpoints/esm2_vicam_650m/CRVDBv29_maxLen1022_Full_lr4e4_RLRP/epoch=9-val_loss=1.40.ckpt'
                state_dict = torch.load(model_checkpoint, weights_only=True)['state_dict']
                new_state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
                self.model.load_state_dict(new_state_dict)


        
        elif self.checkpoint not in supported_models:
            raise ValueError(f"Model {self.checkpoint} not supported. Supported models are: {supported_models}")
        
        
    def setup_model_for_tune(self):
        # freeze all layers but last one
        print("Freezing all layers but the last...")
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.layers[-1].parameters():
            param.requires_grad = True
        
        # Change the lm head to match the number of classes, regression
        self.model_dimension = self.model.embed_dim
        print('Adding a Regression head...')
        self.model.lm_head = RegressionHead(self.model_dimension, self.num_classes, self.hidden_dropout)

        # for name, param in self.model.named_parameters():
        #     if param.requires_grad:
        #         print(f"Trainable parameter: {name}")
        # print()
           

    def get_model_details(self):
        #del self.model.contact_head
        return self.model, self.alphabet



if __name__ == "__main__":
    # Example usage
    args = parse_args()
    model_loader = LoadFromPretrained(args.checkpoint, args.num_classes, args.dropout)
    model, alphabet = model_loader.get_model_details()
    data_module = MyDataModule(model.alphabet, args)
    data_module.setup()
    train_loader = data_module.train_dataloader()

    for batch in train_loader:
        batch_tokens, targets = batch
        preds = model(batch_tokens)
        print(preds)
        break
  