#!/bin/bash
set -e

files=(
    BF520_env_Bloom2018                    H1N1_M1_RF_Wu2016        H3N2_HA_effect_Welsh2024                 OmiXBB15_spike_entry_Dadonaite2024
    BG505_env_Bloom2018                    H1N1_M2_maxRF_Wu2016     H3N2_NA_fitness_Lei2023                  OmiXBB15_spike_escape_Dadonaite2024
    CVB3_capsid_MFElog2_Mattenberger2021   H1N1_M2_RF_Wu2016        HG_FLU_Bloom2016                         PA_FLU_Sun2015
    CVB3_capsid_MFE_Mattenberger2021       H1N1_NEP_enri_Teo2024    HIV1_Rev_SelCoeff_Fernandes2016          POLG_HCVJF_Sun2014
    DENV2_NS5_score_Suphatrakul2023        H1N1_NEP_RF_Teo2024      HIV1_Tat_SelCoeff_Fernandes2016          Zika_env_effect_Sourisseau2019
    H1N1_PA_ddg_Wu2015                     H1N1_M1_maxRF_Wu2016     OmiXBB15_spike_ACE2bind_Dadonaite2024    Zika_env_log2effect_Sourisseau2019
    Delta_spike_observed_Dadonaite2023      Delta_spike_latent_Dadonaite2023                   
    )




# # ViCAM 300M

# MODEL_checkpoint='checkpoints/ViCAM_300M/CRVDBv29_maxLen2046_20aa_Full_RLRP_lr1e6/epoch=9-val_loss=1.52.ckpt'
# version='CRVDBv29_maxLen2046_20aa_Full_RLRP_lr1e6'

MODEL_checkpoint='checkpoints/ViCAM_300M/CRVDBv29_maxLen2046_20aa_Frz3_lr5e4_CALR/epoch=14-val_loss=1.44.ckpt'
version='CRVDBv29_maxLen2046_20aa_Frz3_lr5e4_CALR'

echo "Extracting embedding using model: $MODEL_checkpoint"
for file in "${files[@]}"
do
    echo "Extracting embedding for $file:"
    python scripts/extract_esmc.py -m $MODEL_checkpoint -i "data/DMS_mut_sequences/${file}_muts.fasta"  -o "embeddings/ViCAM/${version}/${file}_embeddings.pt"
done
                




# # ESM C 300M
# MODEL_checkpoint='esmc-300m'
# version=''
# echo "Extracting embedding using model: $MODEL_checkpoint"
# for file in "${files[@]}"
# do
#     echo "Extracting embedding for $file:"
#     python scripts/extract_esmc.py -m $MODEL_checkpoint -i "data/DMS_mut_sequences/${file}_muts.fasta"  -o "embeddings/esmc_300m/${file}_embeddings.pt"
# done





# # ESM-2 3B
# MODEL_checkpoint='rsawhney_esm2_3B'
# version=''

# echo "Extracting embedding using model: $MODEL_checkpoint"
# for file in "${files[@]}"
# do
#     echo "Extracting embedding for $file:"
#     python scripts/extract_esm2_3B_tuned.py -i "data/DMS_mut_sequences/${file}_muts.fasta" -o "embeddings/${MODEL_checkpoint}/${file}_embeddings.pt"
# done