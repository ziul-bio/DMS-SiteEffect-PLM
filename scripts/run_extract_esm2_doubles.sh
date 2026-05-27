#!/bin/bash
set -e

#CUDA_VISIBLE_DEVICES=0 bash scripts/run_extract_esm2_doubles.sh

########################################################################################################
#                                      ProteinGym datasets
########################################################################################################

datasets=(
    A4_HUMAN_Seuma_2022                     NKX31_HUMAN_Tsuboyama_2023_2L9R                 RPC1_BP434_Tsuboyama_2023_1R69
    AMFR_HUMAN_Tsuboyama_2023_4G3O          NUSA_ECOLI_Tsuboyama_2023_1WCL                  SAV1_MOUSE_Tsuboyama_2023_2YSB
    BBC1_YEAST_Tsuboyama_2023_1TG0          NUSG_MYCTU_Tsuboyama_2023_2MI6                  SDA_BACSU_Tsuboyama_2023_1PV0
    BCHB_CHLTE_Tsuboyama_2023_2KRU          OBSCN_HUMAN_Tsuboyama_2023_1V1C                 SPA_STAAU_Tsuboyama_2023_1LP1
    CATR_CHLRE_Tsuboyama_2023_2AMI          ODP2_GEOSE_Tsuboyama_2023_1W4G                  SPG1_STRSG_Olson_2014
    CBPA2_HUMAN_Tsuboyama_2023_1O6X         PABP_YEAST_Melamed_2013                         SPG2_STRSG_Tsuboyama_2023_5UBS
    CBX4_HUMAN_Tsuboyama_2023_2K28          PIN1_HUMAN_Tsuboyama_2023_1I6C                  SPTN1_CHICK_Tsuboyama_2023_1TUD
    CSN4_MOUSE_Tsuboyama_2023_1UFM          PITX2_HUMAN_Tsuboyama_2023_2L7M                 SR43C_ARATH_Tsuboyama_2023_2N88
    CUE1_YEAST_Tsuboyama_2023_2MYX          POLG_PESV_Tsuboyama_2023_2MXD                   SRBS1_HUMAN_Tsuboyama_2023_2O2W
    DLG4_HUMAN_Faure_2021                   PR40A_HUMAN_Tsuboyama_2023_1UZC                 TCRG1_MOUSE_Tsuboyama_2023_1E0L
    DNJA1_HUMAN_Tsuboyama_2023_2LO1         PSAE_PICP2_Tsuboyama_2023_1PSE                  THO1_YEAST_Tsuboyama_2023_2WQG
    DOCK1_MOUSE_Tsuboyama_2023_2M0Y         RAD_ANTMA_Tsuboyama_2023_2CJJ                   TNKS2_HUMAN_Tsuboyama_2023_5JRT
    EPHB2_HUMAN_Tsuboyama_2023_1F0M         RASK_HUMAN_Weng_2022_abundance                  UBE4B_HUMAN_Tsuboyama_2023_3L1X
    FECA_ECOLI_Tsuboyama_2023_2D1U          RASK_HUMAN_Weng_2022_binding-DARPin_K55         UBR5_HUMAN_Tsuboyama_2023_1I2T
    GRB2_HUMAN_Faure_2021                   RBP1_HUMAN_Tsuboyama_2023_2KWH                  VILI_CHICK_Tsuboyama_2023_1YU5
    HECD1_HUMAN_Tsuboyama_2023_3DKM         RCD1_ARATH_Tsuboyama_2023_5OAO                  YAIA_ECOLI_Tsuboyama_2023_2KVT
    ISDH_STAAW_Tsuboyama_2023_2LHR          RCRO_LAMBD_Tsuboyama_2023_1ORC                  YAP1_HUMAN_Araya_2012
    MAFG_MOUSE_Tsuboyama_2023_1K1V          RD23A_HUMAN_Tsuboyama_2023_1IFY                 YNZC_BACSU_Tsuboyama_2023_2JVD
    MBD11_ARATH_Tsuboyama_2023_6ACV         RFAH_ECOLI_Tsuboyama_2023_2LCL                  MYO3_YEAST_Tsuboyama_2023_2BTT          
    RL20_AQUAE_Tsuboyama_2023_1GYZ    
)


########################################################################################################
#                                      Process All Datasets
########################################################################################################

for file in "${datasets[@]}"
do
    echo "Extracting embedding for $file:"

    python scripts/extract_esm2.py esm2_t33_650M_UR50D \
        "data/ProteinGym_fasta/doubles/${file}.fasta" \
        "embeddings/esm2_650m/ProteinGym/doubles/${file}" \
        --repr_layers 33 --include mean
    done


echo "========================================="
echo "        All extractions complete!"
echo "========================================="