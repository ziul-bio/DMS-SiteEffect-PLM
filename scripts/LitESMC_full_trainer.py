import os
import sys
# Add project root to sys.path
sys.path.append(os.path.join(os.path.abspath('..') , 'ESM2ne'))
from Modules.ESMC_full import Load_from_pretrained

import argparse
import pandas as pd

import torch
from torch.optim import AdamW
from torchmetrics.regression import R2Score
from torch.utils.data import Dataset, DataLoader, random_split
torch.set_float32_matmul_precision('medium')
from sklearn.model_selection import train_test_split

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import CSVLogger




#python scripts/LitESMC_full_trainer.py -i data/BLAT_ECOLX_Ranganathan2015.csv -o results/fineTune/test/full --checkpoint_path /stor/work/Wilke/wilkelab/pLMs_checkpoints/ESMC/esmc_300m_2024_12_v0.pth


def parse_args():
    parser = argparse.ArgumentParser(description="Training script arguments.")
    parser.add_argument("-i", "--data", type=str, required=True, help="Path to input CSV file.")
    parser.add_argument("-o", "--output", type=str, required=True, help="Output CSV file path.")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Model checkpoint name or full path.")
    parser.add_argument("--num_classes", type=int, default=1, help="Number of classes (1 for regression).")
    parser.add_argument("--epochs", type=int, default=40, help="Number of epochs.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"), help="Device for training.")
    parser.add_argument("--learning_rate", type=float, default=1e-6, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=1e-8, help="Weight decay.")
    parser.add_argument("--dropout", type=float, default=0.1, help="CLS dropout probability.")
    return parser.parse_args()



class MyDataset(Dataset):
    """This class just loads the data and return a dataset object, returning the sequences and targets.
    Without any tokenization or padding.
    This will be handled later in the collate_fn.
    """
    def __init__(self, data_file):
        self.data = data_file
        self.sequences = [d['sequence'] for _, d in self.data.iterrows()]
        self.targets = [d['target'] for _, d in self.data.iterrows()]

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        sequences = self.sequences[idx]
        target = torch.tensor(self.targets[idx], dtype=torch.float).unsqueeze(-1)
        
        return sequences, target



class MyDataModule(pl.LightningDataModule):
    def __init__(self, tokenizer, args):
        super().__init__()
        self.args = args
        self.tokenizer = tokenizer
        
    def setup(self, stage=None):
        data = pd.read_csv(args.data)
        # sklearn train_test_split
        train_df, val_df = train_test_split(data, test_size=0.2, random_state=42)
        self.train_dataset = MyDataset(train_df)
        self.val_dataset = MyDataset(val_df)

        # dataset = MyDataset(data)
        # # Use random_split to split the dataset into train and validation sets
        # train_size = int(0.8 * len(dataset))
        # val_size = len(dataset) - train_size
        # self.train_dataset, self.val_dataset = random_split(
        #     dataset, [train_size, val_size],
        #     generator=torch.Generator().manual_seed(42)
        #     )
       
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
            drop_last=True,
        )


################ pytorch lightning model ######################
class LitModel(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
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
        return AdamW(self.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)



     



def main():

    # Parse the arguments, and collect the model and dataset names
    args = parse_args()
    model_name = 'esm2_' + args.checkpoint_path.split("/")[-1].split("_")[2]
    print(f"Model name: {model_name}")
    dataset_name = args.data.split("/")[-1].split(".csv")[0]
    print(f"Dataset name: {dataset_name}")


    print("Loading model and dataset...")
    model = LitModel(args)
    datamodule = MyDataModule(model.tokenizer, args)

    
    ########################## Training setup ##########################
    logger = CSVLogger(
            save_dir=os.path.join(f"{args.output}", f"{model_name}"),
            name=f"{dataset_name}",
            version=None,)
        
    trainer = pl.Trainer(
        accelerator="gpu",
        #strategy="ddp",
        devices=1,                  # [0, 1] for 2 GPUs, or -1 for all available GPUs
        #accumulate_grad_batches=4,          # simulate a 4× larger batch size (so 3x4=16)
        max_epochs= args.epochs,
        enable_checkpointing=False,
        gradient_clip_val=1.0,              # Clip gradients if they exceed 1.0
        logger=logger,
        callbacks=[EarlyStopping(monitor="val_loss", patience=5, mode="min")],
    )
    trainer.fit(model, datamodule)

  
  
  

if __name__ == '__main__':
    args = parse_args()
    main()