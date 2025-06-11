#!/bin/bash
set -e

files=(
    'PA_FLU_Sun2015'
    'CVB3_capsid_MFElog2_Mattenberger2021' 'HIV1_Tat_SelCoeff_Fernandes2016'       'OmiXBB15_spike_escape_Dadonaite2024'
    'CVB3_capsid_MFE_Mattenberger2021'     'OmiXBB15_spike_ACE2bind_Dadonaite2024' 'Zika_env_effect_Sourisseau2019'
    'H3N2_NA_fitness_Lei2023'              'OmiXBB15_spike_entry_Dadonaite2024'    'Zika_env_log2effect_Sourisseau2019'
    'BG505_env_Bloom2018'                  'HIV1_Rev_SelCoeff_Fernandes2016'       'PA_FLU_Sun2015'
    'BF520_env_Bloom2018'                  'HG_FLU_Bloom2016'                      'POLG_HCVJF_Sun2014'

    )

MODEL_checkpoint='checkpoints/ViCAM_300M/CRVDBv29_maxLen2046_20aa_Full_lr1e6/epoch=9-val_loss=1.52.ckpt'
version='CRVDBv29_maxLen2046_20aa_Full_lr1e6'

for file in "${files[@]}"
do
    echo "Extracting embedding for $file:"
    python scripts/extract_esmc.py -m $MODEL_checkpoint -i "data/DMS_muts/${file}_muts.fasta"  -o "embeddings/ViCAM/${version}/${file}_embeddings.pt"

done
                