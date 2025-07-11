#!/bin/bash
set -e

# usage
#CUDA_VISIBLE_DEVICES=2 python scripts/LitESMC_partial_trainer.py -i "data/viral/metadata/CVB3_2A_Alvarez.csv" -o "experiments/fineTune/esmc_300m/"  --seed 13 --split_strategy pool_split --checkpoint_path $ESMC_300M_PATH
#CUDA_VISIBLE_DEVICES=3 python scripts/LitESMC_partial_trainer.py -i data/nonviral/metadata/BLAT_ECOLX_Ranganathan2015.csv -o "experiments/fineTune/esmc_300m/"  --seed 13 --split_strategy pool_split --checkpoint_path /stor/work/Wilke/wilkelab/pLMs_checkpoints/ESMC/esmc_300m_2024_12_v0.pth
#bash scripts/run_esmc2ne.sh 

ESMC_300M_PATH=/stor/work/Wilke/wilkelab/pLMs_checkpoints/ESMC/esmc_300m_2024_12_v0.pth
viral=(
    ####################### Debora marks datasets #######################
    # Singles
    IAV_NA_Jiang                  IAV_H1_HA_Wu                  IAV_H1_NP_Doud                   IAV_H5_HA_Dadonaite     
    IAV_H1_HA_Doud                IAV_PA_Wu                     IAV_PB2_Soh                      IAV_RDRP_Li
    IAV_H3_NP_Doud                IAV_H3_HA_Lee
    CVB3_2A_Alvarez               CVB3_2B_Alvarez               CVB3_2C_Alvarez                  CVB3_3A_Alvarez    
    CVB3_3B_Alvarez               CVB3_3C_Alvarez               CVB3_3D_Alvarez                  CVB3_POLG_Mattenberger
    CVB3_VP3_Alvarez              CVB3_VP1_Alvarez
    SARS2_PRD0038_RBD_Starr       SARS2_RBD_binding_Starr       SARS2_RBD_expression_Starr       SARS2_XBB15_RBD_Taylor
    SARS2_BA1_SPIKE_Dadonaite     SARS2_DELTA_SPIKE_Dadonaite   SARS2_MRPO_Flynn                 SARS2_PLPRO_abundance_Wu
    SARS2_PLPRO_activity_Wu
    HIV1_BF520_ENV_Haddox         HIV1_BG505_ENV_Haddox         HIV1_HV1B9_ENV_DuenasDecamp  
    RmYN02_RBD_Starr              RsYN04_RBD_Starr              BP434_RPC1_Tsuboyama             LAMBDA_HCP_Tsuboyama 
    EV_CAPSD_Bakhache             EV_REP_Bakhache               DENV_POLG_Suphatrakul            LASSA_GP_Carr                           
    NIPAH_F_Larsen                             
    ## Doubles and multiple      
    #PESV_POLG_Tsuboyama     BPP22_COAT_Tsuboyama 
    ## Multiple
    #AAV2_CAPSD_Sinai

    )


strategies=('pool_split' 'site_split')
seeds=(13 42 91)

for dts in "${viral[@]}"
do
    for strategy in "${strategies[@]}"
    do
        for sd in "${seeds[@]}"
        do
            echo "Fine-tuning esmc_300m on $dts with strategy $strategy and seed $sd"
            
            CUDA_VISIBLE_DEVICES=2 python scripts/LitESMC_partial_trainer.py \
                -i "data/viral/metadata/${dts}.csv" \
                -o "experiments/fineTune/esmc_300m/" \
                --seed $sd \
                --split_strategy $strategy \
                --checkpoint_path "$ESMC_300M_PATH"

            echo ""
        done
    done
done


