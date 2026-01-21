import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from utils import find_device, find_n_proc
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

class TopKSAE(nn.Module):
    def __init__(self, d_model=64, expansion_factor=64, k=8):
        """
        Args:
            d_model: input dimension - latent dim
            expansion_factor: latent dim expansion factor -> number of features
            k: number of active features (Top K)
        """
        super().__init__()
        
        self.d_model = d_model
        self.d_sae = d_model * expansion_factor
        self.k = k

        # 1. Encoder (W_enc, b_enc)
        self.encoder = nn.Linear(self.d_model, self.d_sae)
        
        # 2. Decoder (W_dec, b_dec)
        # Używamy nn.Parameter dla W_dec, aby łatwiej robić normalizację kolumn
        self.W_dec = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(self.d_sae, self.d_model)))
        self.b_dec = nn.Parameter(torch.zeros(self.d_model))

        # 3. Bias Encodera (często inicjalizowany na 0)
        self.encoder.bias.data.zero_()
        
        # Opcjonalnie: 'Pre-decoder bias' - trik stabilizujący z paperów Anthropic/OpenAI
        # Odejmujemy bias dekodera od wejścia przed enkodowaniem.

    def forward(self, x):
        # x shape: [batch_size, d_model]

        # A. Pre-encoder bias trick (opcjonalne, ale zalecane)
        # Centrujemy wejście względem biasu dekodera
        x_centered = x - self.b_dec

        # B. Encoding
        # Zwykle dajemy ReLU przed TopK, żeby nie aktywować 'anty-cech' (ujemnych korelacji)
        pre_activations = torch.relu(self.encoder(x_centered))

        # C. TopK Selection
        # Wybieramy k największych wartości dla każdego przykładu w batchu
        topk_values, topk_indices = torch.topk(pre_activations, self.k, dim=-1)

        # D. Budowanie rzadkiego wektora (Sparse Feature Vector)
        # Tworzymy tensor zer i wstawiamy w niego tylko wartości TopK
        sparse_acts = torch.zeros_like(pre_activations)
        sparse_acts.scatter_(-1, topk_indices, topk_values)

        # E. Decoding
        # x_hat = (f * W_dec) + b_dec
        x_reconstructed = sparse_acts @ self.W_dec + self.b_dec

        return x_reconstructed, sparse_acts

    @torch.no_grad()
    def normalize_decoder(self):
        """
        Normalizing decoder weights to have Euclidean norm = 1.
        Prevents 'scale hacking'.
        """
        # W_dec shape: [d_sae, d_model]
        # Liczymy normę wzdłuż wymiaru d_model (dim=1)
        norms = torch.norm(self.W_dec, p=2, dim=1, keepdim=True)
        
        # Zabezpieczenie przed dzieleniem przez zero (dodajemy epsilon)
        self.W_dec.div_(norms + 1e-8)


class ActivationsDataset(Dataset):
    def __init__(self, activations_path):
        self.activations = np.load(activations_path, mmap_mode='r', allow_pickle=True)
        self.num_samples, self.seq_len, self.d_model = self.activations.shape
        self.total_vectors = self.num_samples * self.seq_len
        
    def __len__(self):
        return self.total_vectors
    
    def __getitem__(self, idx):
        sample_idx = idx // self.seq_len
        token_idx = idx % self.seq_len
        return torch.from_numpy(self.activations[sample_idx, token_idx].copy()).float()


def _visualize_training(history: dict, window_size: int = 100):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    ax1 = axes[0]
    ax1.plot(history['loss'], alpha=0.3, label='Loss (raw)')

    if len(history['loss']) >= window_size:
        smoothed = np.convolve(history['loss'], np.ones(window_size)/window_size, mode='valid')
        ax1.plot(range(window_size-1, len(history['loss'])), smoothed, label=f'Loss (MA-{window_size})', color='red')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('MSE Loss')
    ax1.set_title('Training Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.plot(history['lr'])
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Learning Rate')
    ax2.set_title('Learning Rate Schedule')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def _evaluate_model(sae: TopKSAE, activations_loader: DataLoader, k: int, device: torch.device):
    sae.eval()

    with torch.no_grad():
        test_batch = next(iter(activations_loader)).to(device)
        recon, sparse_acts = sae(test_batch)
        
        test_loss = F.mse_loss(recon, test_batch)
        
        avg_nonzero = torch.count_nonzero(sparse_acts, dim=-1).float().mean().item()
        
        var_original = torch.var(test_batch).item()
        var_residual = torch.var(test_batch - recon).item()
        explained_var = 1 - (var_residual / var_original)
        
    print(f"Random batch evaluation:")
    print(f"  - MSE Loss: {test_loss.item():.6f}")
    print(f"  - Average number of active features: {avg_nonzero:.1f} (expected: {k})")
    print(f"  - Explained variance: {explained_var*100:.2f}%")


def train_autoencoder(d_model: int, expansion_factor: int, k: int, data_path: str, seq_length: int, layer_num: int, batch_size_sae: int = 256, num_epochs: int = 5, learning_rate: float = 1e-3, save_every_n_steps: int = 10000):
    os.makedirs(os.path.join(data_path, 'models'), exist_ok=True)
    os.makedirs(os.path.join(data_path, 'models/checkpoints'), exist_ok=True)

    activations_path = os.path.join(data_path, f'activations/activations_layer_{layer_num}.npy')
    checkpoint_path = os.path.join(data_path, f'models/checkpoints/topk_sae_layer_{layer_num}.pt')

    device = find_device()

    activations_dataset = ActivationsDataset(activations_path)
    print(f"Number of activation vectors: {len(activations_dataset):,}")
    print(f"Activation vector dimension: {activations_dataset.d_model}")

    activations_loader = DataLoader(
        activations_dataset,
        batch_size=batch_size_sae,
        shuffle=True,
        num_workers=find_n_proc(),
        pin_memory=device.type == 'cuda'
    )
    print(f"Number of batches per epoch: {len(activations_loader):,}")

    sae = TopKSAE(d_model=d_model, expansion_factor=expansion_factor, k=k)
    sae = sae.to(device)

    optimizer = torch.optim.Adam(sae.parameters(), lr=learning_rate)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=num_epochs * len(activations_loader),
        eta_min=1e-5
    )

    print(f"TopK SAE Model configuration:")
    print(f"  - Input dimension: {d_model}")
    print(f"  - Latent dimension (dictionary size): {d_model * expansion_factor}")
    print(f"  - Number of active features (Top K): {k}")
    print(f"  - Number of parameters: {sum(p.numel() for p in sae.parameters()):,}")

    
    # ==================== SAE Training Loop ====================
    history = {
        'loss': [],
        'lr': []
    }

    global_step = 0
    best_loss = float('inf')

    for epoch in range(num_epochs):
        sae.train()
        epoch_losses = []
        
        pbar = tqdm(activations_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_idx, batch in enumerate(pbar):
            x = batch.to(device)  # [batch_size, d_model]
            
            x_reconstructed, sparse_acts = sae(x)
            
            loss = F.mse_loss(x_reconstructed, x)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Important: Normalize decoder weights after each step
            sae.normalize_decoder()
            
            # Update scheduler
            scheduler.step()
            
            epoch_losses.append(loss.item())
            history['loss'].append(loss.item())
            history['lr'].append(scheduler.get_last_lr()[0])
            
            global_step += 1
            
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{scheduler.get_last_lr()[0]:.2e}"
            })
            
            if global_step % save_every_n_steps == 0:
                checkpoint = {
                    'epoch': epoch,
                    'global_step': global_step,
                    'model_state_dict': sae.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': loss.item(),
                    'history': history
                }
                torch.save(checkpoint, checkpoint_path.replace('.pt', f'_step_{global_step}.pt'))
        
        avg_loss = np.mean(epoch_losses)
        print(f"\nEpoch {epoch+1}/{num_epochs} completed - Average loss: {avg_loss:.4f}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': sae.state_dict(),
                'loss': avg_loss,
                'config': {
                    'd_model': d_model,
                    'expansion_factor': expansion_factor,
                    'k': k
                }
            }, checkpoint_path.replace('.pt', '_best.pt'))
            print(f"  -> Saved new best model (loss: {best_loss:.4f})")

    torch.save({
        'epoch': num_epochs,
        'model_state_dict': sae.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history,
        'config': {
            'd_model': d_model,
            'expansion_factor': expansion_factor,
            'k': k
        }
    }, checkpoint_path)

    print(f"\nTraining completed!")
    print(f"Final model saved to: {checkpoint_path}")
    print(f"Best model saved to: {checkpoint_path.replace('.pt', '_best.pt')}")

    _visualize_training(history)

    _evaluate_model(sae, activations_loader, k, device)

    return sae
