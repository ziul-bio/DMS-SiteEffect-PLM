import torch
import torch.nn as nn
from torch.optim import AdamW 
from torch.optim.lr_scheduler import SequentialLR, LinearLR, ConstantLR, CosineAnnealingLR, ReduceLROnPlateau
torch.set_float32_matmul_precision('medium')

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger 
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.tuner.tuning import Tuner
from pytorch_lightning.strategies import DDPStrategy

#import esm
from src.model.ESM2_MLM import Load_from_pretrained

from src.data.dataset_esm2 import MyDataModule
from src.model.esm2_config import config


################ pytorch lightning model ######################
class LitModel(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.model, self.alphabet = Load_from_pretrained(config['model_checkpoint']).get_model_details()

        self.learning_rate = config['learning_rate']
        self.weight_decay = config['weight_decay']
        self.beta1 = config['beta1']
        self.beta2 = config['beta2']
        
        # metrics and loss function
        pad_token_id = self.alphabet.padding_idx
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=pad_token_id)

   
    def forward(self, inputs):
        output = self.model(inputs['input_ids'], return_contacts=False)
        logits = output['logits']
        return logits

    def training_step(self, batch):
        """This function will be called for each batch during training.
        It will compute the loss and log it.
        """
        logits = self(batch)
        targets = batch['labels']
        loss = self.loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss
 
    def validation_step(self, batch):
        logits = self(batch)
        targets = batch['labels']
        loss = self.loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))
        perplexity = torch.exp(loss)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("val_perplexity", perplexity, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss # this loss is not used, but I could return something else and modify.
    
    def test_step(self, batch):
        logits = self(batch)
        targets = batch['labels']
        loss = self.loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))
        perplexity = torch.exp(loss)
        self.log("test_loss", loss, prog_bar=True, logger=True, sync_dist=True)
        self.log("test_perplexity", perplexity, prog_bar=True, logger=True, sync_dist=True)
        return loss, perplexity


    def configure_optimizers(self):
        Optimizer = AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay, betas=(self.beta1, self.beta2))

        LRscheduler = ReduceLROnPlateau(Optimizer, mode='min', factor=0.9, patience=1, cooldown=0) # It decays exponentially, by 10% each time it's triggered.
        
        return {
            "optimizer": Optimizer,
            "lr_scheduler": {"scheduler": LRscheduler, "monitor": "val_loss", "interval": "epoch", "frequency": 1}
            }


def main(args):

    dataset_name = args.dataDir.split('/')[-2]
    print(f"Dataset name: {dataset_name}")
    datamodule = MyDataModule(args)
    print('Loading model...')
    model = LitModel()
 

    ########################### Training setup ##########################
    logger = TensorBoardLogger(
            save_dir='logs/',
            name=args.output,
            version=''
            )
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=f'checkpoints/{args.output}/',  
        filename="{epoch}-{val_loss:.2f}",  
        save_on_train_epoch_end=True,
        monitor="val_loss",               
        save_top_k=5,                      
        mode="min",                             # Mode for monitoring (min for loss, max for accuracy, etc.)
    )
    
    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        patience=5,                            # I am looking for 5 epochs, so patience=5*number of checks per epoch
        mode='min',
        min_delta=0.001,
    )
        
    trainer = pl.Trainer(
        accelerator="gpu",
        num_nodes=1,
        devices=2,                               # [0, 1] for 2 GPUs, or -1 for all available GPUs
        strategy="ddp",
        #strategy='ddp_find_unused_parameters_true',
        #strategy=DDPStrategy(find_unused_parameters=True),
        #accumulate_grad_batches=100,            # simulate a × larger batch size (so 20x4=80) 
	    max_epochs= args.epochs,                 
        enable_checkpointing=True,
        gradient_clip_val=1.0,                   # Clip gradients if they exceed 1.0
        logger=logger,
        callbacks=[early_stopping_callback, checkpoint_callback],
    )
    
    ##################### Finding learning rate #####################
    if args.LRfinder:
        tuner = Tuner(trainer)
        lr_finder = tuner.lr_find(model, datamodule, min_lr=1e-10, max_lr=1e-2, num_training=100)
        fig = lr_finder.plot(suggest=True)                                                                     # type: ignore
        lr_suggestion = round(lr_finder.suggestion(), 10)                                                      # type: ignore
        print(f"Suggested learning rate: {lr_suggestion}")
        fig.savefig(f"logs/{config['output']}/LRfind_{lr_suggestion}.png")                                          # type: ignore

    
    ########################### Training ##########################    
    if args.resume:
        print(f"Resuming training from {args.checkpoint_resume}!")
        trainer.fit(model, datamodule, ckpt_path=args.checkpoint_resume)
        trainer.test(model, datamodule=datamodule)

    elif not args.LRfinder:
        print("Training from scratch!")
        trainer.fit(model, datamodule)
        trainer.test(model, datamodule=datamodule)

    

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-i', '--dataDir', type=str, default='data/processed/test_data/')
    parser.add_argument('-o', '--output', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--LRfinder', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--checkpoint_resume', type=str)
    args = parser.parse_args()
    main(args)




###### RUNNING EXAMPLES ######  
# python src/training/train.py -o vicam_300M/test_lr_finder --LRfinder
# python src/training/train.py -o vicam_300M/CRVDBv29_maxLen2046_Full_lr5e4_RLRP
# python src/training/train.py -o ViCAM_300m/CRVDBv29_maxLen1022_Full_lr5e4_RLRP --resume --checkpoint_resume checkpoints/ViCAM_300m/CRVDBv29_maxLen1022_Full_lr5e4_RLRP/epoch=12-val_loss=1.20.ckpt


###### RUNNING EXAMPLES ######  
# python src/training/train_esm2.py -o esm2_vicam_650m/test_lr_finder --LRfinder
# python src/training/train_esm2.py -o esm2_vicam_650m/CRVDBv29_maxLen1022_Full_test
# python src/training/train_esm2.py -o esm2_vicam_650m/CRVDBv29_maxLen1022_Full_lr4e4_RLRP --resume --checkpoint_resume checkpoints/esm2_vicam_650m/CRVDBv29_maxLen1022_Full_lr4e4_RLRP/epoch=5-val_loss=1.44.ckpt
