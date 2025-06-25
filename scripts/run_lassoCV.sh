#!/bin/bash
set -e

# usage
#taskset -c 50-112 bash run_lassoCV.sh 


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
    echo "Running regression for dataset $dts, using ESM-300M embeddings"                             
    python scripts/lassoCV.py -i "embeddings/esmc_300m/${dts}_embeddings.pt" -m "data/DMS_mut_metadata/${dts}_metadata.csv" -o "experiments/lassoCV/esmc_300m/${dts}.csv"    
    
    echo "Running regression for dataset $dts, using rsawhney_esm2_3B embeddings"                             
    python scripts/lassoCV.py -i "embeddings/rsawhney_esm2_3B/${dts}_embeddings.pt" -m "data/DMS_mut_metadata/${dts}_metadata.csv" -o "experiments/lassoCV/rsawhney_esm2_3B/${dts}.csv"    
    
    echo "Running regression for dataset $dts, using ViCAM-300M embeddings"                             
    python scripts/lassoCV.py -i "embeddings/ViCAM/CRVDBv29_maxLen2046_20aa_Full_RLRP_lr1e6/${dts}_embeddings.pt" -m "data/DMS_mut_metadata/${dts}_metadata.csv" -o "experiments/lassoCV/vicam/CRVDBv29_maxLen2046_20aa_Full_RLRP_lr1e6/${dts}.csv"    
    
    echo "Running regression for dataset $dts, using Frz3 ViCAM-300M embeddings"                             
    python scripts/lassoCV.py -i "embeddings/ViCAM/CRVDBv29_noPoly_Frz3_lr5e4_CALR/${dts}_embeddings.pt" -m "data/DMS_mut_metadata/${dts}_metadata.csv" -o "experiments/lassoCV/vicam/CRVDBv29_noPoly_Frz3_lr5e4_CALR/${dts}.csv"    
    
    echo "Running regression for dataset $dts, using Frz3 ViCAM-300M embeddings"                             
    python scripts/lassoCV.py -i "embeddings/ViCAM/CRVDBv29_maxLen2046_20aa_Frz3_lr5e4_CALR/${dts}_embeddings.pt" -m "data/DMS_mut_metadata/${dts}_metadata.csv" -o "experiments/lassoCV/vicam/CRVDBv29_maxLen2046_20aa_Frz3_lr5e4_CALR/${dts}.csv"    
    
    echo " "                                  
done

