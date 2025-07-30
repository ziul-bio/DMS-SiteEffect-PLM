#!/bin/bash
set -e

# usage
#taskset -c 50-112 bash run_LassoCV.sh 
#python scripts/reg_LassoCV_sitesplit.py -i embeddings/rsawhney_esm2_3B/viral/IAV_PA_Wu_embeddings.pt -m data/viral/metadata/IAV_PA_Wu.csv -o "experiments/lassoCV_SS/rsawhney_esm2_3B/viral/IAV_PA_Wu.csv"    

viral=(
    ####################### Debora marks datasets #######################
    # Singles
    # IAV_NA_Jiang                  IAV_H1_HA_Wu                  IAV_H1_NP_Doud                   IAV_H5_HA_Dadonaite     
    # IAV_H1_HA_Doud                IAV_PA_Wu                     IAV_PB2_Soh                      IAV_RDRP_Li
    # IAV_H3_NP_Doud                IAV_H3_HA_Lee
    # CVB3_2A_Alvarez               CVB3_2B_Alvarez               CVB3_2C_Alvarez                  CVB3_3A_Alvarez    
    # CVB3_3B_Alvarez               CVB3_3C_Alvarez               CVB3_3D_Alvarez                  CVB3_POLG_Mattenberger
    # CVB3_VP3_Alvarez              CVB3_VP1_Alvarez
    # SARS2_PRD0038_RBD_Starr       SARS2_RBD_binding_Starr       SARS2_RBD_expression_Starr       SARS2_XBB15_RBD_Taylor
    # SARS2_BA1_SPIKE_Dadonaite     SARS2_DELTA_SPIKE_Dadonaite   SARS2_MRPO_Flynn                 SARS2_PLPRO_abundance_Wu
    # SARS2_PLPRO_activity_Wu
    # HIV1_BF520_ENV_Haddox         HIV1_BG505_ENV_Haddox         HIV1_HV1B9_ENV_DuenasDecamp  
    # RmYN02_RBD_Starr              RsYN04_RBD_Starr              
    BPP22_COAT_Tsuboyama              
    #LAMBDA_HCP_Tsuboyama 
    #EV_CAPSD_Bakhache             EV_REP_Bakhache               DENV_POLG_Suphatrakul            LASSA_GP_Carr                           
    #NIPAH_F_Larsen                             
    ## Doubles and multiple      
    #PESV_POLG_Tsuboyama     BP434_RPC1_Tsuboyama
    ## Multiple
    #AAV2_CAPSD_Sinai

    )

for dts in "${viral[@]}"
do
    echo "Running regression for dataset $dts, using embeddings"                             
    #python scripts/lassoCV.py -i "embeddings/rsawhney_esm2_3B/viral/${dts}_embeddings.pt" -m "data/viral/metadata/${dts}.csv" -o "experiments/lassoCV/rsawhney_esm2_3B/viral/${dts}.csv"    
    python scripts/reg_LassoCV_sitesplit.py -i "embeddings/rsawhney_esm2_3B/viral/${dts}_embeddings.pt" -m "data/viral/metadata/${dts}.csv" -o "experiments/lassoCV_SS/rsawhney_esm2_3B/viral/${dts}.csv"    
    echo " "                                  
done

