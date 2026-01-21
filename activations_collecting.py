from torch.utils.data import Dataset
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import find_device, find_n_proc
from tqdm import tqdm
import os
from torch.utils.data import DataLoader

class TokenDataset(Dataset):
    def __init__(self, tokens_np_path, seq_length: int):
        self.arr = np.load(tokens_np_path, mmap_mode="r")
        assert self.arr.ndim == 2 and self.arr.shape[1] == seq_length
    def __len__(self):
        return self.arr.shape[0]
    def __getitem__(self, idx):
        x = torch.from_numpy(self.arr[idx]).long()
        return x

def collect_activations(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, data_path: str, seq_length: int, layer_num: int, checkpoint_freq: int = 10000, batch_size: int = 4):
    dataset = TokenDataset(os.path.join(data_path, "sequenced/tokens_seqs_padded.npy"), seq_length)

    os.makedirs(os.path.join(data_path, 'activations'), exist_ok=True)

    # Temporary memmap file for incremental collection
    temp_memmap_file = os.path.join(data_path, f'activations/TEMP_activations_layer_{layer_num}.dat')
    # Final output file (proper .npy format)
    output_file = os.path.join(data_path, f'activations/activations_layer_{layer_num}.npy')
    progress_file = os.path.join(data_path, f'activations/activations_progress_layer_{layer_num}.txt')

    device = find_device()
    model = model.to(device)
    model.eval()

    hidden_size = model.config.hidden_size
    print(f"Hidden size: {hidden_size}")
    print(f"Device: {device}")

    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False,  # no shuffling to be able to resume anytime
        num_workers=find_n_proc(),
        pin_memory=device.type == 'cuda'
    )

    total_samples = len(dataset)
    total_batches = len(dataloader)
    print(f"Total samples: {total_samples}, Total batches: {total_batches}")

    activations_shape = (total_samples, seq_length, hidden_size)

    start_batch = 0
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            start_batch = int(f.read().strip())
        print(f"Resuming from batch {start_batch}")
    else:
        activations_memmap = np.memmap(
            temp_memmap_file, 
            dtype=np.float16, 
            mode='w+', 
            shape=activations_shape
        )
        del activations_memmap
        print(f"Created new temp memmap file: {temp_memmap_file}")


    activations_memmap = np.memmap(
        temp_memmap_file, 
        dtype=np.float16, 
        mode='r+', 
        shape=activations_shape
    )  

    print(f"Shape: {activations_memmap.shape}")

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, initial=start_batch, total=total_batches, desc="Collecting activations")):
            if batch_idx < start_batch:
                continue
            
            input_ids = batch.to(device)
            
            outputs = model(
                input_ids,
                output_hidden_states=True,
                return_dict=True
            )
            
            layer_activations = outputs.hidden_states[layer_num + 1]  # [batch_size, seq_len, hidden_size]
            
            layer_activations_np = layer_activations.cpu().numpy().astype(np.float16)
            
            # calculating indices in memmap
            start_idx = batch_idx * batch_size
            end_idx = start_idx + layer_activations_np.shape[0]
            
            activations_memmap[start_idx:end_idx] = layer_activations_np
            
            if (batch_idx + 1) % checkpoint_freq == 0:
                activations_memmap.flush()
                
                with open(progress_file, 'w') as f:
                    f.write(str(batch_idx + 1))
                
                tqdm.write(f"Saved progress: batch {batch_idx + 1}/{total_batches}")


    activations_memmap.flush()
    with open(progress_file, 'w') as f:
        f.write(str(total_batches))

    # Convert memmap to proper .npy file
    print(f"\nConverting memmap to .npy format...")
    print(f"Size in memory: {activations_memmap.nbytes / 1e9:.2f} GB")
    np.save(output_file, activations_memmap)
    
    # Clean up temp memmap file
    del activations_memmap
    if os.path.exists(temp_memmap_file):
        os.remove(temp_memmap_file)
        print(f"Removed temp memmap file: {temp_memmap_file}")

    print(f"\nFinished! Activations saved to: {output_file}")
    print(f"Shape: {activations_shape}")
    print(f"You can now load with: np.load('{output_file}')")