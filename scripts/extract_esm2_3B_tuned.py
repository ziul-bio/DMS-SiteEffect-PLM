import os
import torch
import argparse
from Bio import SeqIO
from tqdm import tqdm

from transformers import EsmForMaskedLM, AutoTokenizer
from peft import PeftModel, PeftConfig


# Usage:
# python scripts/extract_esm2_3B_tuned.py -i data/DMS_mut_sequences/Delta_spike_latent_Dadonaite2023_muts.fasta -o embeddings/rsawhney_esm2_3B/Delta_spike_latent_Dadonaite2023_embeddings.pt
# python scripts/extract_esm2_3B_tuned.py -i data/DMS_mut_sequences/Delta_spike_observed_Dadonaite2023_muts.fasta -o embeddings/rsawhney_esm2_3B/Delta_spike_observed_Dadonaite2023_embeddings.pt


class FastaDataLoader:
    """
    Data loader for reading a FASTA file and creating batches based on a token limit.
    
    Args:
    - fasta_file (str): Path to the FASTA file.
    - batch_token_limit (int, optional): Maximum number of tokens per batch. Defaults to 4096.
    - model (object): Model object with a `_tokenize` method for tokenizing sequences.
    """
    def __init__(self, fasta_file, batch_token_limit=16000):
        self.fasta_file = fasta_file
        self.batch_token_limit = batch_token_limit
        self.sequences = list(SeqIO.parse(fasta_file, "fasta"))
        self.total_sequences = len(self.sequences)
        
        # Check for duplicate sequence labels
        sequence_labels = [seq.id for seq in self.sequences]
        assert len(set(sequence_labels)) == len(sequence_labels), "Found duplicate sequence labels"

    def __len__(self):
        # Approximate total number of batches
        total_tokens = sum(len(str(seq.seq)) + 2 for seq in self.sequences)  # +2 for BOS and EOS tokens
        return (total_tokens + self.batch_token_limit - 1) // self.batch_token_limit

    def __iter__(self):
        ids, lengths, seqs = [], [], []
        current_token_count = 0

        for seq in self.sequences:
            seq_length = len(seq.seq)
            token_count = seq_length + 2  # Include BOS and EOS tokens
            if current_token_count + token_count > self.batch_token_limit and ids:
                # Yield current batch if adding the new sequence exceeds the token limit
                yield ids, lengths, seqs
                ids, lengths, seqs = [], [], []
                current_token_count = 0

            # Add the current sequence to the batch
            ids.append(seq.id)
            lengths.append(seq_length)
            seqs.append(str(seq.seq))
            current_token_count += token_count

        # Yield any remaining sequences
        if ids:
            yield ids, lengths, seqs



def extract_mean_representations(model, tokenizer, fasta_file):
    mean_representations = {}
    data_loader = FastaDataLoader(fasta_file)
    
    with torch.no_grad():  # Disable gradient calculations
        for batch_ids, batch_lengths, batch_seqs in tqdm(data_loader, desc="Processing batches", leave=False):
            tokens = tokenizer(batch_seqs, return_tensors="pt", padding=True)
            output = model(**tokens.to('cuda'), output_hidden_states=True)

            embeddings = output['hidden_states'][-1] # last layer
    
            # Extract the last hidden states for the sequence
            for i, ID in enumerate(batch_ids):
                representations =  embeddings[i, 1:batch_lengths[i]+1, :].detach().to('cpu') 
                # extract mean representation of the sequence
                mean_representations[ID] = (representations.mean(dim=0))

    return mean_representations


def main():
    parser = argparse.ArgumentParser(description="Extracting ESMC representations from a FASTA file")
    parser.add_argument("-i", "--input_fasta", type=str, required=True, help="Path to the input FASTA file")
    parser.add_argument("-o", "--output", type=str, required=True, help="Path to the output file")
    #parser.add_argument("--gpu", default=0)
    args = parser.parse_args()

    # Define the input parameters
    path_input_fasta_file = args.input_fasta
    output_file = args.output

    # Create the base directory if it doesn't exist
    base_dir = os.path.dirname(output_file)
    if not os.path.exists(base_dir):
        os.makedirs(base_dir, exist_ok=True)


    # Load base model
    base_model = EsmForMaskedLM.from_pretrained('facebook/esm2_t36_3B_UR50D')
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained('facebook/esm2_t36_3B_UR50D')
    # Load the PEFT adapter
    model = PeftModel.from_pretrained(base_model, 'checkpoints/esm2_3B_Sawhney/')
    model.eval()  
    model.to('cuda') 
    

    # Extract representations
    result = extract_mean_representations(model, tokenizer, path_input_fasta_file)
    
    # Save results
    torch.save(result, output_file)
    print(f'Process Finished!')


if __name__ == "__main__":
    main()