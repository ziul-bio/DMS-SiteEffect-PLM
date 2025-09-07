import os
import random
from tqdm import tqdm
from Bio import SeqIO

def split_fasta(input_fasta, output_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42, seq_max_length=1022):
    """
    Splits a FASTA file into train, validation, and test sets.
     Args:
         input_fasta (str): Path to the input FASTA file.
         output_dir (str): Directory to save the split files.
         train_ratio (float): Proportion of sequences for training.
         val_ratio (float): Proportion of sequences for validation.
         test_ratio (float): Proportion of sequences for testing.
         seed (int): Random seed for reproducibility.
     """

    assert train_ratio + val_ratio + test_ratio == 1.0, "Ratios must sum to 1.0"
    print('Reading fasta!')
    records = list(SeqIO.parse(input_fasta, "fasta"))
    random.seed(seed)
    random.shuffle(records)

    valid_amino_acids = set("ACDEFGHIKLMNPQRSTVWY")
    valid_records = []
    longer_seq = 0
    invalid_count = 0 

    for record in tqdm(records, desc="Processing sequences"):
        sequence_str = str(record.seq).upper()
        if len(sequence_str) > seq_max_length:
            print(f"Skipping sequence {record.id} with length {len(sequence_str)} > {seq_max_length}.")
            longer_seq += 1
            continue

        is_valid = True
        for aa in sequence_str:
            if aa not in valid_amino_acids:
                is_valid = False
                print(f"Invalid character '{aa}' found in sequence: {record.id}. Skipping.")
                invalid_count += 1
                break

        if is_valid:
            valid_records.append(record)

    print(f"Total sequences: {len(records)}")
    print(f"Valid sequences: {len(valid_records)}")
    print(f"{longer_seq} sequences were longer than {seq_max_length}.")
    print(f"Invalid characters found in {invalid_count} sequences.")

    # Split based on valid_records
    total = len(valid_records)
    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)

    train_records = valid_records[:train_size]
    val_records = valid_records[train_size:train_size + val_size]
    test_records = valid_records[train_size + val_size:]

    os.makedirs(output_dir, exist_ok=True)
    SeqIO.write(train_records, os.path.join(output_dir, "train.fasta"), "fasta")
    SeqIO.write(val_records, os.path.join(output_dir, "val.fasta"), "fasta")
    SeqIO.write(test_records, os.path.join(output_dir, "test.fasta"), "fasta")

    print(f"Split completed: {len(train_records)} train, {len(val_records)} val, {len(test_records)} test sequences.")


if __name__ == "__main__":
    seq_max_length = 1600 
    input_fasta = "data/raw/URVDBv30prot_rmdup.fasta"
    output_dir = f"data/processed/URVDBv30prot_rmdup_maxlen{seq_max_length}_20aa"
    split_fasta(input_fasta, output_dir, train_ratio=0.9, val_ratio=0.05, test_ratio=0.05, seq_max_length=seq_max_length)
