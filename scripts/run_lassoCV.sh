#!/bin/bash
set -e

# usage
#taskset -c 50-112 bash run_lassoCV.sh 


viral=(
    ####################### Debora marks datasets #######################
    # Singles
    IAV_NA_Jiang                  IAV_H1_HA_Wu                  IAV_H1_NP_Doud                   IAV_H5_HA_Dadonaite     
    IAV_H1_HA_Doud                IAV_PA_Wu                     IAV_PB2_Soh                      IAV_RDRP_Li
    IAV_H3_NP_Doud                IAV_H3_HA_Lee
    CVB3_2A_Alvarez               CVB3_2B_Alvarez               CVB3_2C_Alvarez                  CVB3_3A_Alvarez    
    CVB3_3B_Alvarez               CVB3_3C_Alvarez               CVB3_3D_Alvarez                  CVB3_VP1_Alvarez
    CVB3_VP3_Alvarez              CVB3_POLG_Mattenberger
    
    SARS2_PRD0038_RBD_Starr       SARS2_PLPRO_activity_Wu       SARS2_PLPRO_abundance_Wu         SARS2_MRPO_Flynn
    SARS2_RBD_binding_Starr       SARS2_RBD_expression_Starr    SARS2_BA1_SPIKE_Dadonaite        SARS2_DELTA_SPIKE_Dadonaite   
    SARS2_XBB15_RBD_Taylor                 
    
    HIV1_BF520_ENV_Haddox         HIV1_BG505_ENV_Haddox         HIV1_HV1B9_ENV_DuenasDecamp      NIPAH_F_Larsen
    RmYN02_RBD_Starr              RsYN04_RBD_Starr              DENV_POLG_Suphatrakul            LAMBDA_HCP_Tsuboyama 
    LASSA_GP_Carr                           
    EV_CAPSD_Bakhache             EV_REP_Bakhache               BPP22_COAT_Tsuboyama          
                                 
    ## Doubles and multiple      
    # PESV_POLG_Tsuboyama     BP434_RPC1_Tsuboyama
    ## Multiple
    #AAV2_CAPSD_Sinai
    )

for dts in "${viral[@]}"
do
    # echo "Running regression for dataset $dts, using ESM2-650M embeddings"                             
    # python scripts/lassoCV.py -i "embeddings/esm2_650m/viral/${dts}.pt" -m "data/viral/metadata/${dts}_metadata.csv" -o "experiments/lassoCV/esm2_650m/viral/${dts}.csv"    
    # python scripts/reg_LassoCV_sitesplit.py -i "embeddings/esm2_650m/viral/${dts}.pt" -m "data/viral/metadata/${dts}.csv" -o "experiments/lassoCV_SS/esm2_650m/viral/${dts}.csv"    
    
    echo "Running regression for dataset $dts, using ESM2 Viral 650M embeddings"                             
    python scripts/lassoCV.py -i "embeddings/esm2_viral_650m/viral/${dts}.pt" -m "data/viral/metadata/${dts}.csv" -o "experiments/lassoCV/esm2_viral_650m/viral/${dts}.csv"    
    python scripts/reg_LassoCV_sitesplit.py -i "embeddings/esm2_viral_650m/viral/${dts}.pt" -m "data/viral/metadata/${dts}.csv" -o "experiments/lassoCV_SS/esm2_viral_650m/viral/${dts}.csv"    
    

    #echo "Running regression for dataset $dts, using rsawhney_esm2_3B embeddings"                             
    #python scripts/lassoCV.py -i "embeddings/rsawhney_esm2_3B/${dts}_embeddings.pt" -m "data/viral/metadata/${dts}_metadata.csv" -o "experiments/lassoCV/rsawhney_esm2_3B/${dts}.csv"    
    
    # echo "Running regression for dataset $dts, using ViCAM-300M embeddings"                             
    # python scripts/lassoCV.py -i "embeddings/ViCAM/CRVDBv29_maxLen2046_20aa_Full_RLRP_lr1e6/${dts}_embeddings.pt" -m "data/viral/metadata/${dts}_metadata.csv" -o "experiments/lassoCV/vicam/CRVDBv29_maxLen2046_20aa_Full_RLRP_lr1e6/${dts}.csv"    
    
    echo " "                                  
done

