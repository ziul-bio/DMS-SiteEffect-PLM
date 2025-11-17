#!/bin/bash
set -e

#CUDA_VISIBLE_DEVICES=3 bash scripts/run_extract_esm2.sh

########################################################################################################
#                                      Dataset Definitions
########################################################################################################

viral=(
   ####################### Debora marks datasets #######################
    # Singles
    SARS2_RBD_binding_Starr       SARS2_RBD_expression_Starr    SARS2_BA1_SPIKE_Dadonaite        SARS2_DELTA_SPIKE_Dadonaite   
    SARS2_PRD0038_RBD_Starr       SARS2_PLPRO_activity_Wu       SARS2_PLPRO_abundance_Wu         SARS2_MRPO_Flynn
    SARS2_XBB15_RBD_Taylor                 
    
    IAV_NA_Jiang                  IAV_H1_HA_Wu                  IAV_H1_NP_Doud                   IAV_H5_HA_Dadonaite     
    IAV_H1_HA_Doud                IAV_PA_Wu                     IAV_PB2_Soh                      IAV_RDRP_Li
    IAV_H3_NP_Doud                IAV_H3_HA_Lee
    
    CVB3_2A_Alvarez               CVB3_2B_Alvarez               CVB3_2C_Alvarez                  CVB3_3A_Alvarez    
    CVB3_3B_Alvarez               CVB3_3C_Alvarez               CVB3_3D_Alvarez                  CVB3_VP1_Alvarez
    CVB3_VP3_Alvarez              CVB3_POLG_Mattenberger
    
    HIV1_BF520_ENV_Haddox         HIV1_BG505_ENV_Haddox         HIV1_HV1B9_ENV_DuenasDecamp      NIPAH_F_Larsen
    RmYN02_RBD_Starr              RsYN04_RBD_Starr              DENV_POLG_Suphatrakul            LAMBDA_HCP_Tsuboyama 
    LASSA_GP_Carr                 EV_CAPSD_Bakhache             EV_REP_Bakhache                  BPP22_COAT_Tsuboyama          
                                 
    ## Doubles and multiple      
    # PESV_POLG_Tsuboyama     BP434_RPC1_Tsuboyama
    ## Multiple
    #AAV2_CAPSD_Sinai
)

cellular=(
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



########################################################################################################
#                                      Extraction Functions
########################################################################################################

extract_embeddings() {
    local model_name=$1
    local model_checkpoint=$2
    local source=$3
    local datasets=("${!4}")
    local script=$5
    local extra_args="${6:-}"
    
    echo "Extracting embeddings using model: $model_checkpoint"
    for file in "${datasets[@]}"
    do
        echo "Extracting embedding for $file:"
        if [ "$script" == "extract_esm2.py" ]; then
            python scripts/extract_esm2.py esm2_t33_650M_UR50D \
                "data/${source}/mutant_sequences/${file}.fasta" \
                "embeddings/${model_name}/${source}/${file}" \
                --repr_layers 33 --include mean $extra_args
        else
            python scripts/extract_esm2_3B_tuned.py \
                -i "data/${source}/mutant_sequences/${file}.fasta" \
                -o "embeddings/${model_name}/${source}/${file}.pt"
        fi
    done
}

########################################################################################################
#                                      Process All Datasets
########################################################################################################

# Define models
declare -A models=(
    ["esm2_650M_CRVDB"]="checkpoints/esm2_viral_650m/CRVDBv30_partial_MP15/epoch=9-val_loss=1.44.ckpt"
    ["esm2_650M_URVDB"]="checkpoints/esm2_viral_650m/URVDBv30_partial_MP15/epoch=9-val_loss=0.45.ckpt"
    ["esm2_650M"]="esm2_650M"
    ["esm2_3B_Sawhney"]="esm2_3B_Sawhney"
)

# Process each source (viral and cellular)
for source in "viral" "cellular"; do
    echo "========================================="
    echo "Processing $source sequences"
    echo "========================================="
    
    # Get the appropriate dataset
    if [ "$source" == "viral" ]; then
        datasets=("${viral[@]}")
    else
        datasets=("${cellular[@]}")
    fi
    
    # Extract with ESM-2 650M CRVDB
    extract_embeddings "esm2_650M_CRVDB" \
        "${models[esm2_650M_CRVDB]}" \
        "$source" \
        datasets[@] \
        "extract_esm2.py" \
        "--resume --tuned_checkpoint ${models[esm2_650M_CRVDB]}"
    
    # Extract with ESM-2 650M URVDB
    extract_embeddings "esm2_650M_URVDB" \
        "${models[esm2_650M_URVDB]}" \
        "$source" \
        datasets[@] \
        "extract_esm2.py" \
        "--resume --tuned_checkpoint ${models[esm2_650M_URVDB]}"
    
    # Extract with ESM-2 650M (base)
    extract_embeddings "esm2_650M" \
        "${models[esm2_650M]}" \
        "$source" \
        datasets[@] \
        "extract_esm2.py"
    
    # Extract with ESM-2 3B
    extract_embeddings "esm2_3B_Sawhney" \
        "${models[esm2_3B_Sawhney]}" \
        "$source" \
        datasets[@] \
        "extract_esm2_3B_tuned.py"
done

echo "========================================="
echo "All extractions complete!"
echo "========================================="