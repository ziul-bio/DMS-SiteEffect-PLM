import torch
import torch.nn as nn
from torch.optim import AdamW 
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger 
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

# my modules
from src.model.ESMC import Load_from_pretrained
from src.data.dataset import MyDataModule
torch.set_float32_matmul_precision('medium')
from pytorch_lightning.tuner.tuning import Tuner


################ pytorch lightning model ######################
class LitModel(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.model= Load_from_pretrained(args.checkpoint_path).get_model_details()
        self.learning_rate = args.learning_rate
        self.weight_decay = args.weight_decay
        self.beta1 = args.beta1
        self.beta2 = args.beta2
        
        # metrics and loss function
        pad_token_id = 1
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=pad_token_id)
   
    def forward(self, tokens):
        outputs = self.model(tokens['input_ids']) 
        logits, embeddings, hidden_states = outputs.sequence_logits, outputs.embeddings, outputs.hidden_states
        return logits


    def training_step(self, batch):
        """This function will be called for each batch during training.
        It will compute the loss and log it.
        """
        logits = self(batch)
        targets = batch['labels']
        loss = self.loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))
        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True, logger=True, sync_dist=True)
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
        LRscheduler = {
            #'scheduler': CosineAnnealingLR(Optimizer, T_max=10, eta_min=self.learning_rate*0.1),
            'scheduler': ReduceLROnPlateau(Optimizer, mode='min', factor=0.9, patience=1, cooldown=0),
            'interval': 'epoch',  
            'frequency': 1,       
            'monitor': 'val_loss', # only for ReduceLROnPlateau
        }
        return {"optimizer": Optimizer, "lr_scheduler": LRscheduler}    
        #return Optimizer
        




def main(args):
    dataset_name = args.dataDir.split('/')[-2]
    print(f"Dataset name: {dataset_name}")
    datamodule = MyDataModule(args)

    print("Loading model...")
    model = LitModel(args)
    

    ########################### Training setup ##########################
    logger = TensorBoardLogger(
            save_dir='logs/',
            name=args.output,
            version=''
            )
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=f'checkpoints/{args.output}/',  
        filename="{epoch}-{val_loss:.2f}",  
        monitor="val_loss",               
        save_top_k=2,                      
        mode="min",                         # Mode for monitoring (min for loss, max for accuracy, etc.)
    )
    
    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        patience=50,                        # I am looking for 5 epochs, so patience=5*number of checks per epoch
        mode='min',
        min_delta=0.001,
    )
        
    trainer = pl.Trainer(
        accelerator="gpu",
        #strategy="ddp",
        devices=1, #[0, 1, 2, 3],          # [0, 1] for 2 GPUs, or -1 for all available GPUs
        #accumulate_grad_batches=4,          # simulate a 4× larger batch size (so 3x4=16)
        max_epochs= args.epochs,
        max_steps=200000,                    
        val_check_interval=50000,
        enable_checkpointing=True,
        gradient_clip_val=1.0,               # Clip gradients if they exceed 1.0
        logger=logger,
        callbacks=[early_stopping_callback, checkpoint_callback],
    )
    
    ##################### Finding learning rate #####################
    if args.LRfinder:
        tuner = Tuner(trainer)
        lr_finder = tuner.lr_find(model, datamodule, min_lr=1e-8, max_lr=1e-3, num_training=100)
        fig = lr_finder.plot(suggest=True)
        lr_suggestion = round(lr_finder.suggestion(), 10)
        print(f"Suggested learning rate: {lr_suggestion}")
        fig.savefig(f"logs/{args.output}/LRfind_{lr_suggestion}.png")

    
    ########################### Training ##########################    
    if args.resume:
        print(f"Resuming training from {args.checkpoint}!")
        trainer.fit(model, datamodule, ckpt_path=args.checkpoint)
        trainer.test(model, datamodule=datamodule)

    elif not args.LRfinder:
        print("Training from scratch!")
        trainer.fit(model, datamodule)
        trainer.test(model, datamodule=datamodule)

    

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-i', '--dataDir', type=str, default='data/processed/C-RVDBv29_maxlen2046/')
    #parser.add_argument('-i', '--dataDir', type=str, default='data/processed/example/')
    parser.add_argument('-o', '--output', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--learning_rate', type=float, default=5e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.95)
    parser.add_argument('--checkpoint_path', type=str, default='/stor/work/Wilke/wilkelab/pLMs_checkpoints/ESMC/esmc_300m_2024_12_v0.pth')
    parser.add_argument('--LRfinder', action='store_true')
    args = parser.parse_args()
    main(args)




###### RUNNING EXAMPLES ######  
# # this version I reduced the weight decay to 1e-2  
# CUDA_VISIBLE_DEVICES=2 python src/training/train.py -o ViCAM_300M/test_5e4_RLRP 