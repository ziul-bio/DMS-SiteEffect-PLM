#!/bin/bash
set -e

# usage
#bash run_finetune.sh 

CUDA_VISIBLE_DEVICES=3

viral=(
    BF520_env_Bloom2018                    H1N1_M1_RF_Wu2016        H3N2_HA_effect_Welsh2024                 OmiXBB15_spike_entry_Dadonaite2024
    BG505_env_Bloom2018                    H1N1_M2_maxRF_Wu2016     H3N2_NA_fitness_Lei2023                  OmiXBB15_spike_escape_Dadonaite2024
    CVB3_capsid_MFElog2_Mattenberger2021   H1N1_M2_RF_Wu2016        HG_FLU_Bloom2016                         PA_FLU_Sun2015
    CVB3_capsid_MFE_Mattenberger2021       H1N1_NEP_enri_Teo2024    HIV1_Rev_SelCoeff_Fernandes2016          POLG_HCVJF_Sun2014
    DENV2_NS5_score_Suphatrakul2023        H1N1_NEP_RF_Teo2024      HIV1_Tat_SelCoeff_Fernandes2016          Zika_env_effect_Sourisseau2019
    H1N1_PA_ddg_Wu2015                     H1N1_M1_maxRF_Wu2016     OmiXBB15_spike_ACE2bind_Dadonaite2024    Zika_env_log2effect_Sourisseau2019
    Delta_spike_observed_Dadonaite2023     Delta_spike_latent_Dadonaite2023
                              
    )

for dts in "${viral[@]}"
do
    echo "Fine-tuning vicam_300m on $dts:"                             
    python scripts/LitESMC_partial_trainer.py -i "data/DMS_mut_metadata/${dts}_metadata.csv" -o "experiments/fineTune/esmc_300m/${dts}.csv" --checkpoint_path checkpoints/vicam_300m/CRVDBv29_maxLen2046_20aa_Full_RLRP_lr1e6/epoch=9-val_loss=1.52.ckpt
    echo " "                                  
done

