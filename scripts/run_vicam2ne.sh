#!/bin/bash
set -e

# usage
#bash run_finetune.sh 
#IAV_NA_Jiang not working
viral=(
    # BF520_env_Bloom2018                    H1N1_M1_RF_Wu2016        H3N2_HA_effect_Welsh2024                 OmiXBB15_spike_entry_Dadonaite2024
    # BG505_env_Bloom2018                    H1N1_M2_maxRF_Wu2016     H3N2_NA_fitness_Lei2023                  OmiXBB15_spike_escape_Dadonaite2024
    # CVB3_capsid_MFElog2_Mattenberger2021   H1N1_M2_RF_Wu2016        HG_FLU_Bloom2016                         PA_FLU_Sun2015
    # CVB3_capsid_MFE_Mattenberger2021       H1N1_NEP_enri_Teo2024    HIV1_Rev_SelCoeff_Fernandes2016          
    # DENV2_NS5_score_Suphatrakul2023        H1N1_NEP_RF_Teo2024      HIV1_Tat_SelCoeff_Fernandes2016          Zika_env_effect_Sourisseau2019
    # H1N1_PA_ddg_Wu2015                     H1N1_M1_maxRF_Wu2016     OmiXBB15_spike_ACE2bind_Dadonaite2024    Zika_env_log2effect_Sourisseau2019
    # Delta_spike_observed_Dadonaite2023     Delta_spike_latent_Dadonaite2023
    # #POLG_HCVJF_Sun2014

    # CVB3_POLG_Mattenberger  AAV2_CAPSD_Sinai             IAV_H3_HA_Lee         RmYN02_RBD_Starr
    # CVB3_VP1_Alvarez        BP434_RPC1_Tsuboyama         IAV_H3_NP_Doud        RsYN04_RBD_Starr
    # CVB3_VP3_Alvarez        DENV_POLG_Suphatrakul        IAV_H5_HA_Dadonaite   SARS2_BA1_SPIKE_Dadonaite
    # CVB3_2A_Alvarez         EV_CAPSD_Bakhache            IAV_NA_Jiang          SARS2_DELTA_SPIKE_Dadonaite
    # CVB3_2B_Alvarez         EV_REP_Bakhache              IAV_PA_Wu             SARS2_MRPO_Flynn
    # CVB3_2C_Alvarez         HIV1_BF520_ENV_Haddox        IAV_PB2_Soh           SARS2_PLPRO_Wu_abundance
    # CVB3_3A_Alvarez         HIV1_BG505_ENV_Haddox        IAV_RDRP_Li           SARS2_PLPRO_Wu_activity
    # CVB3_3B_Alvarez         HIV1_HV1B9_ENV_DuenasDecamp  IAV_H1_NP_Doud        SARS2_PRD0038_RBD_Starr
    # CVB3_3C_Alvarez         LASSA_GP_Carr                IAV_H1_HA_Doud        SARS2_RBD_Starr_binding
    # CVB3_3D_Alvarez         NIPAH_F_Larsen               IAV_H1_HA_Wu          SARS2_RBD_Starr_expression
    # BPP22_COAT_Tsuboyama    LAMBDA_HCP_Tsuboyama         PESV_POLG_Tsuboyama   SARS2_XBB15_RBD_Taylor   
    CVB3_POLG_Mattenberger                                                        
    )

for dts in "${viral[@]}"
do
    echo "Fine-tuning vicam_300m on $dts:"                             
    #CUDA_VISIBLE_DEVICES=3 python scripts/LitESMC_partial_trainer.py -i "data/viral_dms/${dts}_dms.csv" -o "experiments/fineTune/vicam_300m/" --checkpoint_path checkpoints/vicam_300m/CRVDBv29_maxLen2046_20aa_Full_RLRP_lr1e6/epoch=9-val_loss=1.52.ckpt
    CUDA_VISIBLE_DEVICES=3 python scripts/LitESMC_partial_trainer.py -i "data/viral_dms/${dts}_dms.csv" -o "experiments/fineTune/vicam_300m/" --batch_size 32 --checkpoint_path checkpoints/vicam_300m/CRVDBv29_maxLen2046_20aa_Full_RLRP_lr1e6/epoch=9-val_loss=1.52.ckpt
    echo " "                                  
done

