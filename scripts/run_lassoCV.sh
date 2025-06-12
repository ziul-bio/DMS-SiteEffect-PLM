#!/bin/bash
set -e

# usage
#taskset -c 50-112 bash run_LassoCV.sh 


viral=(
    'PA_FLU_Sun2015'
    'CVB3_capsid_MFElog2_Mattenberger2021' 'HIV1_Tat_SelCoeff_Fernandes2016'       'OmiXBB15_spike_escape_Dadonaite2024'
    'CVB3_capsid_MFE_Mattenberger2021'     'OmiXBB15_spike_ACE2bind_Dadonaite2024' 'Zika_env_effect_Sourisseau2019'
    'H3N2_NA_fitness_Lei2023'              'OmiXBB15_spike_entry_Dadonaite2024'    'Zika_env_log2effect_Sourisseau2019'
    'BG505_env_Bloom2018'                  'HIV1_Rev_SelCoeff_Fernandes2016'       'PA_FLU_Sun2015'
    'BF520_env_Bloom2018'                  'HG_FLU_Bloom2016'                      'POLG_HCVJF_Sun2014'
    )

for dts in "${viral[@]}"
do
    echo "Running regression for dataset $dts, using ESM-300M embeddings"                             
    python scripts/lassoCV.py -i "embeddings/esmc_300m/${dts}_embeddings.pt" -m "data/metadata/${dts}_metadata.csv" -o "experiments/esmc_300m/lassoCV/${dts}.csv"    
    
    echo "Running regression for dataset $dts, using ViCAM-300M embeddings"                             
    python scripts/lassoCV.py -i "embeddings/ViCAM/CRVDBv29_maxLen2046_20aa_Full_lr1e6/${dts}_embeddings.pt" -m "data/metadata/${dts}_metadata.csv" -o "experiments/vicam/lassoCV/CRVDBv29_maxLen2046_20aa_Full_lr1e6/${dts}.csv"    
    echo " "                                  
done