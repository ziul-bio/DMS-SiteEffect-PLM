import os
import sys
from ESMC_partial import Load_from_pretrained

import argparse
import numpy as np
import pandas as pd
import random

import torch
from torch.optim import AdamW
from torchmetrics.regression import R2Score
from torch.utils.data import Dataset, DataLoader, random_split
torch.set_float32_matmul_precision('medium')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.tuner.tuning import Tuner



#python scripts/LitESMC_partial_trainer.py -i data/DMS_mut_metadata/HG_FLU_Bloom2016_metadata.csv -o experiments/fineTune/esmc-300m/partial --checkpoint_path /stor/work/Wilke/wilkelab/pLMs_checkpoints/ESMC/esmc_300m_2024_12_v0.pth
#python scripts/LitESMC_partial_trainer.py -i data/nonviral/metadata/BLAT_ECOLX_Ranganathan2015_metadata.csv -o experiments/fineTune/esmc-300m/partial --checkpoint_path /stor/work/Wilke/wilkelab/pLMs_checkpoints/ESMC/esmc_300m_2024_12_v0.pth
#python scripts/LitESMC_partial_trainer.py -i data/DMS_mut_metadata/HG_FLU_Bloom2016_metadata.csv -o experiments/fineTune/vicam-300m/partial --checkpoint_path checkpoints/vicam_300m/CRVDBv29_maxLen2046_20aa_Full_RLRP_lr1e6/epoch=9-val_loss=1.52.ckpt

#python scripts/LitESMC_partial_trainer.py -i data/DMS_mut_metadata/OmiXBB15_spike_ACE2bind_Dadonaite2024_metadata.csv -o experiments/fineTune/esmc-300m/ --checkpoint_path /stor/work/Wilke/wilkelab/pLMs_checkpoints/ESMC/esmc_300m_2024_12_v0.pth
#python scripts/LitESMC_partial_trainer.py -i data/DMS_mut_metadata/OmiXBB15_spike_ACE2bind_Dadonaite2024_metadata.csv -o experiments/fineTune/esmc-600m/ --checkpoint_path /stor/work/Wilke/wilkelab/pLMs_checkpoints/ESMC/esmc_600m_2024_12_v0.pth

# CUDA_VISIBLE_DEVICES=3 python scripts/LitESMC_partial_trainer.py -i data/nonviral/metadata/BLAT_ECOLX_Ranganathan2015.csv -o results/fineTune/test/partial/esmc-300m  --seed 13 --split_strategy pool_split --checkpoint_path /stor/work/Wilke/wilkelab/pLMs_checkpoints/ESMC/esmc_300m_2024_12_v0.pth

def parse_args():
    parser = argparse.ArgumentParser(description="Training script arguments.")
    parser.add_argument("-i", "--data", type=str, required=True, help="Path to input CSV file.")
    parser.add_argument("-o", "--output", type=str, required=True, help="Output CSV file path.")
    parser.add_argument("--split_strategy", type=str, default="pool_split", help="Split strategy to use.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Model checkpoint name or full path.")
    parser.add_argument("--num_classes", type=int, default=1, help="Number of classes (1 for regression).")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"), help="Device for training.")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay.")
    parser.add_argument("--dropout", type=float, default=0.2, help="CLS dropout probability.")
    return parser.parse_args()



class MyDataset(Dataset):
    """This class just loads the data and return a dataset object, returning the sequences and targets.
    Without any tokenization or padding.
    This will be handled later in the collate_fn.
    """
    def __init__(self, data_file):
        self.scaler = StandardScaler()
        self.data = data_file
        self.sequences = self.data['sequence'].tolist()
        self.targets = self.scaler.fit_transform(self.data['target'].to_frame()).squeeze()

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        sequences = self.sequences[idx]
        target = torch.tensor(self.targets[idx], dtype=torch.float).unsqueeze(-1)
        
        return sequences, target


def split_data(df, seed, train_pct=0.8, val_pct=0.2):
    """
    This function randomly splits a dataframe into train, validation data given a seed by mutation site.
    Parameters:
     - df (DataFrame): dataframe containing information about mutants. Mutants should be in the order wt amino acid, site of mutation, mutant amino acid. ex "M1F"
     - seed (int): the seed to be used when shuffling sites randomly.
     - train_pct (float): the percentage of data that will be split into the train dataset. Default is 0.8
     - val_pct (float): the percentage of data that will be split into the validation dataset. Default is 0.2

    Returns:
     - train_df (DataFrame): the DataFrame containing selected data by site to be used as the train dataset.
     - val_df (DataFrame): the DataFrame containing selected data by site to be used as the validation dataset.
    """
    # find sites of mutation and order randomly
    df["site"] = [int(s[1:-1]) for s in df["mutant"]]
    sites = df["site"].unique()
    random.seed(seed)
    random.shuffle(sites)

    if train_pct + val_pct != 1:
        print("Split percentages must sum to 1")
        return

    df_size = df.shape[0]
    df_val_size = df_size*val_pct
    val_sites, train_sites = [], []

    # determine sites for validation, then train
    for site in sites:
        if len(val_sites) <= df_val_size:
            val_sites.extend([mut_site for mut_site in df["site"] if mut_site == site])
        else:
            train_sites.extend([mut_site for mut_site in df["site"] if mut_site == site])

    # subset df for train, val data
    train_df = df[df["site"].isin(set(train_sites))]
    val_df = df[df["site"].isin(set(val_sites))]
    return train_df, val_df


class MyDataModule(pl.LightningDataModule):
    def __init__(self, tokenizer, args):
        super().__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.split_strategy = args.split_strategy # 'pool_split' or 'site_split'
        self.seed = args.seed 

    def setup(self, stage=None):
        data = pd.read_csv(args.data)
        if self.split_strategy == 'site_split':
            train_df, val_df = split_data(data, seed=self.seed, train_pct=0.8, val_pct=0.2)
        elif self.split_strategy == 'pool_split':
            train_df, val_df = train_test_split(data, random_state=self.seed, test_size=0.2)

        self.train_dataset = MyDataset(train_df)
        self.val_dataset = MyDataset(val_df)
       
    def collate_fn(self, batch):
        """This function will be used to collate the data into a batch.
        It will handle the tokenization and padding of the sequences.
        """
        # batch: a list of ( (ID, sequence), target )
        seqs = [item[0] for item in batch]
        targets = [item[1] for item in batch]

        # Tokenize the sequences
        tokens = self.tokenizer(seqs)
        targets = torch.stack(targets) # Stack the targets to a shape (batch_size, 1)
        return tokens, targets

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
            num_workers=0,
            #persistent_workers=True,
            pin_memory=False,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.args.batch_size,
            collate_fn=self.collate_fn,
            num_workers=0,
            pin_memory=False,
            #persistent_workers=True,
            drop_last=True,
        )


################ pytorch lightning model ######################
class LitModel(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.learning_rate = args.learning_rate
        self.weight_decay = args.weight_decay
        # Define the model
        self.model_loader = Load_from_pretrained(
            checkpoint_path=args.checkpoint_path, num_classes=args.num_classes, dropout=args.dropout)
        self.model, self.tokenizer = self.model_loader.get_model_details()
        
        # metrics and loss function
        self.loss_fn = torch.nn.MSELoss(reduction='mean')
        self.train_r2 = R2Score()
        self.val_r2 = R2Score()
    

    def forward(self, tokens):
        outputs = self.model(tokens) # type: ignore
        return outputs.sequence_logits 


    def training_step(self, batch):
        """This function will be called for each batch during training.
        It will compute the loss and log it.
        """
        tokens, targets = batch
        preds = self(tokens)
        loss = self.loss_fn(preds, targets)
        self.train_r2(preds, targets)
        # Log the loss by epoch
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("train_r2", self.train_r2, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss
 
    def validation_step(self, batch):
        tokens, targets = batch
        preds = self.forward(tokens)
        loss = self.loss_fn(preds, targets)
        self.val_r2(preds, targets)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("val_r2", self.val_r2, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss # this loss is not used, but I could return something else and modify.

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)



     



def main():

    # Parse the arguments, and collect the model and dataset names
    args = parse_args()
    model_name = [x for x in ['vicam_300m', 'vicam_600m','esmc_300m', 'esmc_600m'] if x in args.checkpoint_path][0]
    print(f"Model name: {model_name}")
    dataset_name = args.data.split("/")[-1].split(".csv")[0]
    print(f"Dataset name: {dataset_name}")


    print("Loading model and dataset...")
    model = LitModel(args)
    datamodule = MyDataModule(model.tokenizer, args)

    
    ########################## Training setup ##########################
    logger = CSVLogger(
            #save_dir=os.path.join(f"{args.output}", f"{model_name}"),
            save_dir=os.path.join(f"{args.output}"),
            name=f"{dataset_name}",
            version=f'{args.split_strategy}/seed_{args.seed}',)

    trainer = pl.Trainer(
        accelerator="gpu",
        #strategy="ddp",
        devices=1,                           # [0, 1] for 2 GPUs, or -1 for all available GPUs
        #accumulate_grad_batches=4,          # simulate a 4× larger batch size (so 3x4=16)
        max_epochs= args.epochs,
        enable_checkpointing=False,
        gradient_clip_val=1.0,               # Clip gradients if they exceed 1.0
        logger=logger,
        callbacks=[EarlyStopping(monitor="val_loss", patience=10, mode="min")],
    )     
    
    trainer.fit(model, datamodule)

  
  
  

if __name__ == '__main__':
    args = parse_args()
    main()