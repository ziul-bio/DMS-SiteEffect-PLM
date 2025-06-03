#!/bin/bash
set -e

files=(
    'BF520_env_Bloom2018' 'BG505_env_Bloom2018' 'HG_FLU_Bloom2016'  
    'PA_FLU_Sun2015' 'POLG_HCVJF_Sun2014'
    )

MODEL_checkpoint='checkpoints/logs/ViCAM_300M/CRVDBv29_noPoly_Frz3_lr5e4_RLRP_v02/epoch=28-val_loss=1.38.ckpt'
version='CRVDBv29_noPoly_Frz3_lr5e4_RLRP_v02'

for file in "${files[@]}"
do
    echo "Extracting embedding for $file:"
    python scripts/extract_esmc.py -m $MODEL_checkpoint -i "data/DMS_muts/${file}_muts.fasta"  -o "embeddings/ViCAM/${version}/${file}_embeddings.pt"

done
                