#!/bin/bash
set -e

#python scripts/extract_esmc.py -m checkpoints/vicam_300m/CRVDBv29_maxLen1022_Full_lr5e4_RLRP/epoch=4-val_loss=1.35.ckpt -i "data/DMS_mut_sequences/PA_FLU_Sun2015_muts.fasta"  -o embeddings/ViCAM/CRVDBv29_maxLen1022_Full_lr5e4_RLRP/PA_FLU_Sun2015_embeddings.pt
#rsync -avP checkpoints/ViCAM_300M/CRVDBv29_maxLen1022_Full_lr5e4_RLRP/epoch=4-val_loss=1.35.ckpt lcv454@wilkcomp01.ccbb.utexas.edu:/stor/work/Wilke/luiz/ViCAM/checkpoints/vicam_300m/CRVDBv29_maxLen1022_Full_lr5e4_RLRP/


########################################################################################################
#                                      Viral sequences
########################################################################################################

viral=(
    ####################### Debora marks datasets #######################
    # Singles
    IAV_NA_Jiang                  IAV_H1_HA_Wu                  IAV_H1_NP_Doud                   IAV_H5_HA_Dadonaite     
    IAV_H1_HA_Doud                IAV_PA_Wu                     IAV_PB2_Soh                      IAV_RDRP_Li
    IAV_H3_NP_Doud                IAV_H3_HA_Lee
    
    CVB3_2A_Alvarez               CVB3_2B_Alvarez               CVB3_2C_Alvarez                  CVB3_3A_Alvarez    
    CVB3_3B_Alvarez               CVB3_3C_Alvarez               CVB3_3D_Alvarez                  CVB3_VP1_Alvarez
    CVB3_VP3_Alvarez              CVB3_POLG_Mattenberger
    
    SARS2_PRD0038_RBD_Starr       SARS2_PLPRO_activity_Wu         SARS2_PLPRO_abundance_Wu         SARS2_MRPO_Flynn
    SARS2_RBD_binding_Starr       SARS2_RBD_expression_Starr      SARS2_BA1_SPIKE_Dadonaite        SARS2_DELTA_SPIKE_Dadonaite   
    SARS2_XBB15_RBD_Taylor                 
    
    HIV1_BF520_ENV_Haddox         HIV1_BG505_ENV_Haddox         HIV1_HV1B9_ENV_DuenasDecamp      NIPAH_F_Larsen
    RmYN02_RBD_Starr              RsYN04_RBD_Starr              BPP22_COAT_Tsuboyama             LAMBDA_HCP_Tsuboyama 
    DENV_POLG_Suphatrakul         LASSA_GP_Carr                 EV_CAPSD_Bakhache                EV_REP_Bakhache              
                                 
    ## Doubles and multiple      
    # PESV_POLG_Tsuboyama     BP434_RPC1_Tsuboyama
    ## Multiple
    #AAV2_CAPSD_Sinai
    )


# # ViCAM 600M
# MODEL_checkpoint='checkpoints/vicam_300m/CRVDBv29_maxLen1022_Full_lr5e4_RLRP/epoch=4-val_loss=1.35.ckpt'
# version='CRVDBv29_maxLen1022_Full_lr5e4_RLRP'

# echo "Extracting embedding using model: $MODEL_checkpoint"
# for file in "${viral[@]}"
# do
#     echo "Extracting embedding for $file:"
#     python scripts/extract_esmc.py -m $MODEL_checkpoint -i "data/viral/mutant_sequences/${file}.fasta"  -o "embeddings/esmc_viral_600m/${file}.pt"
# done
                

# ESM C 600M
MODEL_checkpoint='esmc-600m'
version=''
echo "Extracting embedding using model: $MODEL_checkpoint"
for file in "${viral[@]}"
do
    echo "Extracting embedding for $file:"
    python scripts/extract_esmc.py -m $MODEL_checkpoint -i "data/viral/mutant_sequences/${file}.fasta"  -o "embeddings/esmc_600m/viral/${file}.pt"
done







########################################################################################################
#                                      Non-Viral sequences
########################################################################################################




nonviral=(
    ####################### Riesselman 2018, singles. #######################
    'AMIE_PSEAE_Whitehead'            'DLG4_RAT_Ranganathan2012'            'RL401_YEAST_Bolon2014'
    'B3VI55_LIPSTSTABLE'              'GAL4_YEAST_Shendure2015'             'RL401_YEAST_Fraser2016'
    'B3VI55_LIPST_Whitehead2015'      'HSP82_YEAST_Bolon2016'               'SUMO1_HUMAN_Roth2017'
    'BG_STRSQ_hmmerbit'               'IF1_ECOLI'                           'TIM_SULSO'
    'BLAT_ECOLX_Ostermeier2014'       'KKA2_KLEPN_Mikkelsen2014'            'TIM_THEMA'
    'BLAT_ECOLX_Palzkill2012'         'MK01_HUMAN_Johannessen'              'TIM_THETH'
    'BLAT_ECOLX_Ranganathan2015'      'MTH3_HAEAESTABILIZED_Tawfik2015'     'TPK1_HUMAN_Roth2017'
    'BLAT_ECOLX_Tenaillon2013'        'PABP_YEAST_Fields2013_singles'       'TPMT_HUMAN_Fowler2018'
    'BRCA1_HUMAN_BRCT'                'PTEN_HUMAN_Fowler2018'               'UBC9_HUMAN_Roth2017'
    'BRCA1_HUMAN_RING'                'RASH_HUMAN_Kuriyan'                  'UBE4B_MOUSE_Klevit2013_singles'
    'CALM1_HUMAN_Roth2017'            'RL401_YEAST_Bolon2013'               'YAP1_HUMAN_Fields2012_singles'                         
    )



# ESM C 600M
MODEL_checkpoint='esmc-600m'
version=''
echo "Extracting embedding using model: $MODEL_checkpoint"
for file in "${nonviral[@]}"
do
    echo "Extracting embedding for $file:"
    python scripts/extract_esmc.py -m $MODEL_checkpoint -i "data/nonviral/mutant_sequences/${file}.fasta"  -o "embeddings/esmc_600m/nonviral/${file}.pt"
done