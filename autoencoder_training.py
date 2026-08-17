"""Top-K sparse autoencoder and resumable, memory-bounded training."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from utils import find_device, set_seed


class TopKSAE(nn.Module):
    """A tied-dimension Top-K SAE with decoder-direction normalization."""

    def __init__(self, d_model: int = 64, expansion_factor: int = 64, k: int = 8):
        super().__init__()
        if d_model < 1 or expansion_factor < 1:
            raise ValueError("d_model and expansion_factor must be positive")
        d_sae = int(d_model * expansion_factor)
        if not 1 <= k <= d_sae:
            raise ValueError(f"k must be in [1, {d_sae}], got {k}")

        self.d_model = int(d_model)
        self.d_sae = d_sae
        self.k = int(k)
        self.encoder = nn.Linear(self.d_model, self.d_sae)
        self.W_dec = nn.Parameter(torch.empty(self.d_sae, self.d_model))
        self.b_dec = nn.Parameter(torch.zeros(self.d_model))
        nn.init.kaiming_uniform_(self.W_dec, a=np.sqrt(5))
        nn.init.zeros_(self.encoder.bias)
        self.normalize_decoder()

    def forward(self, x: torch.Tensor):
        if x.shape[-1] != self.d_model:
            raise ValueError(f"Expected input dimension {self.d_model}, got {x.shape[-1]}")
        centered = x - self.b_dec
        pre_acts = F.relu(self.encoder(centered))
        values, indices = torch.topk(pre_acts, self.k, dim=-1)
        sparse_acts = torch.zeros_like(pre_acts)
        sparse_acts.scatter_(-1, indices, values)
        reconstruction = sparse_acts @ self.W_dec + self.b_dec
        return reconstruction, sparse_acts

    @torch.no_grad()
    def normalize_decoder(self):
        # Each row is one decoder direction: [d_sae, d_model].
        norms = self.W_dec.norm(p=2, dim=1, keepdim=True).clamp_min(1e-8)
        self.W_dec.div_(norms)

    def config(self) -> Dict[str, int]:
        return {
            "d_model": self.d_model,
            "d_sae": self.d_sae,
            "expansion_factor": self.d_sae // self.d_model,
            "k": self.k,
        }


def _atomic_torch_save(payload: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _trim_history(history: Dict[str, list], max_history: int) -> Dict[str, list]:
    if max_history < 1:
        return {key: [] for key in history}
    return {key: values[-max_history:] for key, values in history.items()}


def _cleanup_step_checkpoints(checkpoint_dir: Path, prefix: str, keep_last: int) -> None:
    if keep_last < 0:
        raise ValueError("keep_last_checkpoints must be non-negative")
    paths = sorted(
        checkpoint_dir.glob(f"{prefix}_step_*.pt"),
        key=lambda path: path.stat().st_mtime,
    )
    for path in paths[:-keep_last] if keep_last else paths:
        path.unlink(missing_ok=True)


def _load_checkpoint(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _checkpoint_payload(
    sae: TopKSAE,
    optimizer: torch.optim.Optimizer,
    scheduler,
    config: Dict,
    history: Dict[str, list],
    epoch: int,
    next_sequence: int,
    global_step: int,
    best_loss: float,
    usage_counts: torch.Tensor,
    status: str,
) -> Dict:
    return {
        "format": "topk_sae_streaming_v2",
        "status": status,
        "config": config,
        "model_state_dict": sae.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "history": history,
        "epoch": int(epoch),
        "next_sequence": int(next_sequence),
        "global_step": int(global_step),
        "best_loss": float(best_loss),
        "usage_counts": usage_counts.cpu(),
        "rng_state": {
            "torch": torch.get_rng_state(),
            "numpy": np.random.get_state(),
            "python": random.getstate(),
        },
    }


def _restore_rng(payload: Dict) -> None:
    state = payload.get("rng_state") or {}
    if state.get("torch") is not None:
        torch.set_rng_state(state["torch"])
    if state.get("numpy") is not None:
        np.random.set_state(state["numpy"])
    if state.get("python") is not None:
        random.setstate(state["python"])


def _count_valid_tokens(mask_path: Path, start: int, end: int, chunk_sequences: int) -> int:
    masks = np.load(mask_path, mmap_mode="r")
    total = 0
    for chunk_start in range(start, end, chunk_sequences):
        chunk_end = min(chunk_start + chunk_sequences, end)
        total += int(np.asarray(masks[chunk_start:chunk_end]).sum())
    return total


def _count_sae_steps(
    mask_path: Path,
    start: int,
    end: int,
    chunk_sequences: int,
    batch_size_sae: int,
) -> int:
    """Count optimizer updates using the same chunk boundaries as training."""
    masks = np.load(mask_path, mmap_mode="r")
    total_steps = 0
    for chunk_start in range(start, end, chunk_sequences):
        chunk_end = min(chunk_start + chunk_sequences, end)
        valid_tokens = int(np.asarray(masks[chunk_start:chunk_end]).sum())
        total_steps += (valid_tokens + batch_size_sae - 1) // batch_size_sae
    return total_steps


def train_autoencoder_streaming(
    model: torch.nn.Module,
    data_path: str,
    seq_length: int,
    layer_num: int,
    d_model: Optional[int] = None,
    expansion_factor: int = 16,
    k: int = 64,
    batch_size_sae: int = 1024,
    model_batch_size: int = 4,
    chunk_sequences: int = 32,
    num_epochs: int = 2,
    learning_rate: float = 1e-3,
    min_learning_rate: float = 1e-5,
    seed: int = 0,
    checkpoint_path: str | None = None,
    resume: bool = True,
    save_every_n_chunks: int = 1,
    keep_last_checkpoints: int = 3,
    max_history: int = 10000,
    max_sequences: Optional[int] = None,
    max_steps: Optional[int] = None,
    device: Optional[torch.device] = None,
    model_name: Optional[str] = None,
    verbose: str = "high",
    verbose_interval: int = 1000,
) -> TopKSAE:
    """Train SAE directly from bounded model-inference chunks.

    A checkpoint is written after each completed chunk by default.  The
    checkpoint cursor points to the next sequence range, so an interruption
    never requires retaining the entire model activation dataset.
    """
    if num_epochs < 1 or batch_size_sae < 1 or save_every_n_chunks < 1:
        raise ValueError("num_epochs, batch_size_sae and save_every_n_chunks must be positive")
    if verbose not in {"low", "high"}:
        raise ValueError("verbose must be 'low' or 'high'")
    if verbose_interval < 1:
        raise ValueError("verbose_interval must be positive")
    set_seed(seed)
    device = device or find_device()
    model = model.to(device).eval()
    inferred_d_model = int(model.config.hidden_size)
    if d_model is None:
        d_model = inferred_d_model
    if int(d_model) != inferred_d_model:
        raise ValueError(f"SAE d_model={d_model} does not match model hidden_size={inferred_d_model}")

    from activations_collecting import open_token_store, iter_activation_chunks

    store = open_token_store(data_path, seq_length)
    effective_end = store.n_sequences if max_sequences is None else min(store.n_sequences, max_sequences)
    if effective_end < 1:
        raise ValueError("No sequences available for SAE training")
    valid_tokens = _count_valid_tokens(store.mask_path, 0, effective_end, chunk_sequences) if store.mask_path.exists() else None
    if valid_tokens is not None and valid_tokens < 1:
        raise ValueError("No valid tokens found in the attention mask")

    config = {
        "data_path": str(Path(data_path).resolve()),
        "seq_length": int(seq_length),
        "layer_num": int(layer_num),
        "d_model": int(d_model),
        "expansion_factor": int(expansion_factor),
        "k": int(k),
        "batch_size_sae": int(batch_size_sae),
        "model_batch_size": int(model_batch_size),
        "chunk_sequences": int(chunk_sequences),
        "num_epochs": int(num_epochs),
        "learning_rate": float(learning_rate),
        "min_learning_rate": float(min_learning_rate),
        "seed": int(seed),
        "max_sequences": None if max_sequences is None else int(max_sequences),
        "model_hidden_size": inferred_d_model,
        "model_name": model_name,
    }
    checkpoint_dir = Path(data_path) / "models" / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(checkpoint_path) if checkpoint_path else checkpoint_dir / f"topk_sae_layer_{layer_num}.pt"
    best_checkpoint = checkpoint.with_name(checkpoint.stem + "_best.pt")
    step_prefix = checkpoint.stem

    sae = TopKSAE(d_model=d_model, expansion_factor=expansion_factor, k=k).to(device)
    optimizer = torch.optim.AdamW(sae.parameters(), lr=learning_rate, weight_decay=0.0)
    if valid_tokens is None:
        steps_per_epoch = max(1, (store.n_sequences * seq_length + batch_size_sae - 1) // batch_size_sae)
    else:
        # Chunks are shuffled and batched independently. Therefore the exact
        # count is sum(ceil(chunk_tokens / batch_size_sae)), not one ceil over
        # all tokens.
        steps_per_epoch = max(
            1,
            _count_sae_steps(
                store.mask_path,
                0,
                effective_end,
                chunk_sequences,
                batch_size_sae,
            ),
        )
    total_steps = max(1, num_epochs * steps_per_epoch)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=min_learning_rate
    )

    history = {"loss": [], "lr": [], "epoch_loss": [], "active_features": []}
    start_epoch = 0
    next_sequence = 0
    global_step = 0
    best_loss = float("inf")
    usage_counts = torch.zeros(sae.d_sae, dtype=torch.long)

    if resume and checkpoint.exists():
        payload = _load_checkpoint(checkpoint)
        old_config = payload.get("config", {})
        for key in config:
            if old_config.get(key) != config.get(key):
                raise ValueError(
                    f"Checkpoint configuration mismatch for {key}: "
                    f"checkpoint={old_config.get(key)!r}, requested={config.get(key)!r}"
                )
        sae.load_state_dict(payload["model_state_dict"])
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        scheduler.load_state_dict(payload["scheduler_state_dict"])
        history = _trim_history(payload.get("history", history), max_history)
        start_epoch = int(payload.get("epoch", 0))
        next_sequence = int(payload.get("next_sequence", 0))
        global_step = int(payload.get("global_step", 0))
        best_loss = float(payload.get("best_loss", float("inf")))
        if "usage_counts" in payload:
            usage_counts = payload["usage_counts"].long()
        _restore_rng(payload)
        print(f"Resuming SAE from epoch={start_epoch}, next_sequence={next_sequence}, step={global_step}")

    print(
        f"Training plan: {num_epochs} epoch(s), {steps_per_epoch:,} SAE steps/epoch, "
        f"{total_steps:,} expected optimizer steps"
    )
    progress = None
    if verbose == "high":
        progress = tqdm(
            total=total_steps,
            initial=min(global_step, total_steps),
            desc="SAE optimizer steps",
            unit="step",
        )

    processed_steps = global_step
    stop_training = False
    resume_epoch = start_epoch
    for epoch in range(start_epoch, num_epochs):
        epoch_start_sequence = next_sequence if epoch == start_epoch else 0
        epoch_losses = []
        chunk_counter = 0
        iterator = iter_activation_chunks(
            model=model,
            data_path=data_path,
            seq_length=seq_length,
            layer_num=layer_num,
            batch_size=model_batch_size,
            chunk_sequences=chunk_sequences,
            start_sequence=epoch_start_sequence,
            max_sequences=effective_end - epoch_start_sequence,
            device=device,
            output_dtype=np.float16,
        )
        chunk_total = max(
            1,
            (effective_end - epoch_start_sequence + chunk_sequences - 1) // chunk_sequences,
        )
        chunk_progress = None
        if verbose == "high":
            chunk_progress = tqdm(
                total=chunk_total,
                desc=f"Epoch {epoch + 1}/{num_epochs} activation chunks",
                unit="chunk",
                leave=False,
            )
        for chunk_start, chunk_end, activations in iterator:
            if chunk_progress is not None:
                chunk_progress.update(1)
            # max_steps is enforced at chunk boundaries. This keeps the
            # checkpoint cursor exact without needing to persist a partially
            # shuffled chunk and its local batch offset.
            if max_steps is not None and processed_steps >= max_steps:
                stop_training = True
                break
            if activations.shape[0] == 0:
                next_sequence = chunk_end
                continue
            x = torch.from_numpy(activations).float()
            # Deterministic local shuffling avoids a huge global permutation.
            generator = torch.Generator(device="cpu")
            generator.manual_seed(seed + epoch * 1_000_003 + chunk_start)
            order = torch.randperm(x.shape[0], generator=generator)
            chunk_loss_sum = 0.0
            chunk_count = 0
            sae.train()
            for batch_start in range(0, x.shape[0], batch_size_sae):
                batch_indices = order[batch_start:batch_start + batch_size_sae]
                batch = x[batch_indices].to(device, non_blocking=True)
                reconstruction, sparse_acts = sae(batch)
                loss = F.mse_loss(reconstruction, batch)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(sae.parameters(), max_norm=1.0)
                optimizer.step()
                sae.normalize_decoder()
                scheduler.step()

                loss_value = float(loss.detach().cpu())
                chunk_loss_sum += loss_value * batch.shape[0]
                chunk_count += int(batch.shape[0])
                epoch_losses.append(loss_value)
                usage_counts += torch.count_nonzero(
                    sparse_acts.detach(), dim=0
                ).cpu().long()
                history["loss"].append(loss_value)
                history["lr"].append(float(scheduler.get_last_lr()[0]))
                history["active_features"].append(int(torch.count_nonzero(sparse_acts).item()))
                global_step += 1
                processed_steps += 1
                if progress is not None:
                    progress.update(1)
                    progress.set_postfix(
                        loss=f"{loss_value:.5f}",
                        lr=f"{scheduler.get_last_lr()[0]:.2e}",
                    )
                elif (
                    processed_steps % verbose_interval == 0
                    or (max_steps is not None and processed_steps >= max_steps)
                ):
                    print(
                        f"SAE progress: step {processed_steps:,}/{total_steps:,}; "
                        f"loss={loss_value:.6f}; "
                        f"lr={scheduler.get_last_lr()[0]:.2e}"
                    )
                if max_steps is not None and processed_steps >= max_steps:
                    stop_training = True

            next_sequence = chunk_end
            chunk_counter += 1
            history = _trim_history(history, max_history)
            if stop_training or chunk_counter % save_every_n_chunks == 0:
                payload = _checkpoint_payload(
                    sae, optimizer, scheduler, config, history, epoch,
                    next_sequence, global_step, best_loss, usage_counts,
                    "paused" if stop_training else "running",
                )
                step_path = checkpoint_dir / f"{step_prefix}_step_{global_step}.pt"
                _atomic_torch_save(payload, step_path)
                _atomic_torch_save(payload, checkpoint)
                _cleanup_step_checkpoints(checkpoint_dir, step_prefix, keep_last_checkpoints)
            if stop_training:
                break

        if chunk_progress is not None:
            chunk_progress.close()

        if stop_training:
            if next_sequence >= effective_end:
                resume_epoch = epoch + 1
                next_sequence = 0
            else:
                resume_epoch = epoch
            paused_payload = _checkpoint_payload(
                sae, optimizer, scheduler, config, history, resume_epoch,
                next_sequence, global_step, best_loss, usage_counts, "paused",
            )
            _atomic_torch_save(paused_payload, checkpoint)
            break
        avg_loss = float(np.mean(epoch_losses)) if epoch_losses else float("inf")
        history["epoch_loss"].append(avg_loss)
        history = _trim_history(history, max_history)
        next_sequence = 0
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_payload = _checkpoint_payload(
                sae, optimizer, scheduler, config, history, epoch + 1,
                0, global_step, best_loss, usage_counts, "best",
            )
            _atomic_torch_save(best_payload, best_checkpoint)
        final_payload = _checkpoint_payload(
            sae, optimizer, scheduler, config, history, epoch + 1,
            0, global_step, best_loss, usage_counts, "running",
        )
        _atomic_torch_save(final_payload, checkpoint)
        print(f"Epoch {epoch + 1}/{num_epochs}: mean_loss={avg_loss:.6f}, step={global_step}")

    final_payload = _checkpoint_payload(
        sae, optimizer, scheduler, config, history,
        (epoch + 1 if not stop_training else resume_epoch),
        next_sequence, global_step, best_loss, usage_counts,
        "paused" if stop_training else "complete",
    )
    _atomic_torch_save(final_payload, checkpoint)
    if progress is not None:
        progress.close()
    if not stop_training:
        dead_fraction = float((usage_counts == 0).float().mean())
        print(f"SAE training complete: checkpoint={checkpoint}, dead_feature_fraction={dead_fraction:.4%}")
    else:
        print(f"SAE training paused at step {global_step}; rerun with --resume to continue")
    return sae


# Compatibility path for pre-existing full activation files. New Pythia runs
# should use train_autoencoder_streaming instead.
def train_autoencoder(
    d_model: int, expansion_factor: int, k: int, data_path: str, seq_length: int,
    layer_num: int, batch_size_sae: int = 256, num_epochs: int = 5,
    learning_rate: float = 1e-3, save_every_n_steps: int = 10000,
):
    activation_path = Path(data_path) / f"activations/activations_layer_{layer_num}.npy"
    if not activation_path.exists():
        raise FileNotFoundError(
            f"Legacy activation file not found: {activation_path}. "
            "Use train_autoencoder_streaming for storage-bounded training."
        )
    activations = np.load(activation_path, mmap_mode="r")
    if activations.ndim != 3 or activations.shape[-1] != d_model:
        raise ValueError(f"Expected activation array [N,S,{d_model}], got {activations.shape}")
    device = find_device()
    sae = TopKSAE(d_model=d_model, expansion_factor=expansion_factor, k=k).to(device)
    optimizer = torch.optim.AdamW(sae.parameters(), lr=learning_rate)
    total_steps = num_epochs * ((activations.shape[0] * activations.shape[1] + batch_size_sae - 1) // batch_size_sae)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max(1, total_steps), eta_min=1e-5)
    checkpoint_dir = Path(data_path) / "models" / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"topk_sae_layer_{layer_num}.pt"
    history = {"loss": [], "lr": []}
    for epoch in range(num_epochs):
        indices = np.random.default_rng(epoch).permutation(activations.shape[0] * activations.shape[1])
        losses = []
        for start in range(0, len(indices), batch_size_sae):
            flat = indices[start:start + batch_size_sae]
            seq = flat // activations.shape[1]
            pos = flat % activations.shape[1]
            x = torch.from_numpy(np.asarray(activations[seq, pos])).float().to(device)
            recon, sparse = sae(x)
            loss = F.mse_loss(recon, x)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            sae.normalize_decoder()
            scheduler.step()
            losses.append(float(loss.detach().cpu()))
            history["loss"].append(losses[-1])
            history["lr"].append(float(scheduler.get_last_lr()[0]))
        print(f"Epoch {epoch + 1}/{num_epochs}: mean_loss={np.mean(losses):.6f}")
    payload = {
        "format": "topk_sae_legacy_v2",
        "config": sae.config() | {"seq_length": seq_length, "layer_num": layer_num},
        "model_state_dict": sae.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "history": history,
        "epoch": num_epochs,
    }
    _atomic_torch_save(payload, checkpoint_path)
    return sae
