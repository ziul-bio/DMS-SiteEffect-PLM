import os
import torch
from typing import List

import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from Bio import SeqIO

import esm
# Load ESM-2 model
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
batch_converter = alphabet.get_batch_converter()

class MLM:
    """
    ## Masked LM (MLM)

    This class implements the masking procedure for a given batch of token sequences.
    """

    def __init__(self, *,
                 padding_token: int, mask_token: int, no_mask_tokens: List[int], n_tokens: int,
                 masking_prob: float = 0.15, randomize_prob: float = 0.1, no_change_prob: float = 0.1,
                 ):
        """
        * `padding_token` is the padding token `[PAD]`.
          We will use this to mark the labels that shouldn't be used for loss calculation.
        * `mask_token` is the masking token `[MASK]`.
        * `no_mask_tokens` is a list of tokens that should not be masked.
        This is useful if we are training the MLM with another task like classification at the same time,
        and we have tokens such as `[CLS]` that shouldn't be masked.
        * `n_tokens` total number of tokens (used for generating random tokens)
        * `masking_prob` is the masking probability
        * `randomize_prob` is the probability of replacing with a random token
        * `no_change_prob` is the probability of replacing with original token
        """
        self.n_tokens = n_tokens
        self.no_change_prob = no_change_prob
        self.randomize_prob = randomize_prob
        self.masking_prob = masking_prob
        self.no_mask_tokens = no_mask_tokens + [padding_token, mask_token]
        self.padding_token = padding_token
        self.mask_token = mask_token

    def __call__(self, x: torch.Tensor):
        """
        * `x` is the batch of input token sequences.
         It's a tensor of type `long` with shape `[seq_len, batch_size]`.
        """

        # Mask `masking_prob` of tokens
        full_mask = torch.rand(x.shape, device=x.device) < self.masking_prob
        # Unmask `no_mask_tokens`
        for t in self.no_mask_tokens:
            full_mask &= x != t

        # A mask for tokens to be replaced with original tokens
        unchanged = full_mask & (torch.rand(x.shape, device=x.device) < self.no_change_prob)
        # A mask for tokens to be replaced with a random token
        random_token_mask = full_mask & (torch.rand(x.shape, device=x.device) < self.randomize_prob)
        # Indexes of tokens to be replaced with random tokens
        random_token_idx = torch.nonzero(random_token_mask, as_tuple=True)
        # Random tokens for each of the locations
        random_tokens = torch.randint(0, self.n_tokens, (len(random_token_idx[0]),), device=x.device)
        # The final set of tokens that are going to be replaced by `[MASK]`
        mask = full_mask & ~random_token_mask & ~unchanged

        # Make a clone of the input for the labels
        y = x.clone()

        # Replace with `[MASK]` tokens;
        # note that this doesn't include the tokens that will have the original token unchanged and
        # those that get replace with a random token.
        x.masked_fill_(mask, self.mask_token)
        # Assign random tokens
        x[random_token_idx] = random_tokens

        # Assign token `[PAD]` to all the other locations in the labels.
        # The labels equal to `[PAD]` will not be used in the loss.
        y.masked_fill_(~full_mask, self.padding_token)

        # Return the masked input and the labels
        return x, y




class MyDataset(Dataset):
    """This function collates a batch, tokenizes and pads the sequences,
    and applies the MLM masking strategy to produce inputs and targets.
    """
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequences = self.sequences[idx]

        return sequences
    

class MyDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.data_folder = args.dataDir
        self.batch_size = args.batch_size

        self.mlm = MLM(
            padding_token=alphabet.padding_idx,
            mask_token=alphabet.mask_idx,
            no_mask_tokens=[alphabet.cls_idx, alphabet.eos_idx],
            n_tokens=len(alphabet.all_toks),
            masking_prob=0.15,
            randomize_prob=0.1,
            no_change_prob=0.1
        )
    

    def setup(self, stage=None):
        """This method will load the data splits preprocessed."""

        with open(f"{self.data_folder}/train.fasta", "r") as f:
            train_sequences = [(str(record.id), str(record.seq)) for record in SeqIO.parse(f, "fasta")]
            self.train_dataset =MyDataset(train_sequences) 

        with open(f"{self.data_folder}/val.fasta", "r") as f:
            val_sequences = [(str(record.id), str(record.seq)) for record in SeqIO.parse(f, "fasta")]
            self.val_dataset = MyDataset(val_sequences)

        with open(f"{self.data_folder}/test.fasta", "r") as f:
            test_sequences = [(str(record.id), str(record.seq)) for record in SeqIO.parse(f, "fasta")]
            self.test_dataset = MyDataset(test_sequences)
    
            
    
    def bert_collate_fn(self, batch):
        """This function will be used to collate (collect and combine) the data into a batch.
        It will handle the tokenization and padding of the sequences.
        And also creates the target sequence shifted by one, in relation to the input sequence.
        """
        # Tokenize the batch of sequences
        batch_labels, batch_strs, batch_tokens = batch_converter(batch)

        x, y = self.mlm(batch_tokens)
       
        
        return {
            'input_ids': x,
            'labels': y,
        }

    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.bert_collate_fn,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            collate_fn=self.bert_collate_fn,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            collate_fn=self.bert_collate_fn,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
        )
    

if __name__ == "__main__":
    print("Running test on the dataset module")
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--dataDir', type=str, default='data/RVDB/processed/CRVDBv30prot_maxlen1600_20aa')
    parser.add_argument('--batch_size', type=int, default=1)
    args = parser.parse_args()

    data_module = MyDataModule(args)
    data_module.setup()
    train_loader = data_module.train_dataloader()

    print()
    for batch in train_loader:
        input_ids = batch["input_ids"][0]
        labels = batch["labels"][0]
        print(f'Input sequence: {input_ids}')
        print(f'Target sequence: {labels}')
        break