#!/bin/bash
set -e

#python scripts/extract_esmc.py -m checkpoints/vicam_300m/CRVDBv29_maxLen1022_Full_lr5e4_RLRP/epoch=4-val_loss=1.35.ckpt -i "data/DMS_mut_sequences/PA_FLU_Sun2015_muts.fasta"  -o embeddings/ViCAM/CRVDBv29_maxLen1022_Full_lr5e4_RLRP/PA_FLU_Sun2015_embeddings.pt
#rsync -avP checkpoints/ViCAM_300M/CRVDBv29_maxLen1022_Full_lr5e4_RLRP/epoch=4-val_loss=1.35.ckpt lcv454@wilkcomp01.ccbb.utexas.edu:/stor/work/Wilke/luiz/ViCAM/checkpoints/vicam_300m/CRVDBv29_maxLen1022_Full_lr5e4_RLRP/

files=(
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


# # ViCAM 300M
# MODEL_checkpoint='checkpoints/vicam_300m/CRVDBv29_maxLen1022_Full_lr5e4_RLRP/epoch=4-val_loss=1.35.ckpt'
# version='CRVDBv29_maxLen1022_Full_lr5e4_RLRP'

# echo "Extracting embedding using model: $MODEL_checkpoint"
# for file in "${files[@]}"
# do
#     echo "Extracting embedding for $file:"
#     python scripts/extract_esmc.py -m $MODEL_checkpoint -i "data/DMS_mut_sequences/${file}_muts.fasta"  -o "embeddings/ViCAM/${version}/${file}_embeddings.pt"
# done
                





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
MODEL_checkpoint='rsawhney_esm2_3B'
version=''

echo "Extracting embedding using model: $MODEL_checkpoint"
for file in "${files[@]}"
do
    echo "Extracting embedding for $file:"
    CUDA_VISIBLE_DEVICE=3 python scripts/extract_esm2_3B_tuned.py -i "data/viral/mutante_sequences/${file}_muts.fasta" -o "embeddings/${MODEL_checkpoint}/viral/${file}_embeddings.pt"
done