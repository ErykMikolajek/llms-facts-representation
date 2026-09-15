from multiprocessing import cpu_count
import hashlib
import os
import random

import numpy as np
import torch


def find_n_proc():
    return max(cpu_count() - 1, 1)


def tokenizer_vocab_fingerprint(tokenizer) -> str:
    """Stable hash guarding token-id semantics across saved artifacts."""
    vocabulary = tokenizer.get_vocab()
    digest = hashlib.sha256()
    for token, token_id in sorted(vocabulary.items(), key=lambda item: (item[1], item[0])):
        digest.update(str(int(token_id)).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(token).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _cuda_kernel_supported() -> bool:
    """Check whether this PyTorch build contains kernels for the active GPU."""
    try:
        major, minor = torch.cuda.get_device_capability()
        device_arch = f"sm_{major}{minor}"
        compiled_arches = set(torch.cuda.get_arch_list())
    except Exception:
        # Let the normal CUDA path handle unusual/older PyTorch builds.
        return True
    return not compiled_arches or device_arch in compiled_arches


def find_device():
    """Select a usable device, avoiding CUDA wheels incompatible with the GPU.

    ``SAE_DEVICE=cpu`` is useful on Kaggle when a P100 is paired with a recent
    PyTorch wheel that only contains ``sm_70+`` kernels.
    """
    if os.environ.get("SAE_DEVICE", "auto").lower() == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        if _cuda_kernel_supported():
            return torch.device("cuda")
        capability = torch.cuda.get_device_capability()
        compiled = ", ".join(torch.cuda.get_arch_list()) or "unknown"
        print(
            "CUDA GPU capability "
            f"sm_{capability[0]}{capability[1]} is not supported by this "
            f"PyTorch build ({compiled}); falling back to CPU. "
            "Use a T4/L4/A100 Kaggle accelerator for GPU training."
        )
        return torch.device("cpu")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    """Set all local RNGs used by sequencing and SAE training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available() and _cuda_kernel_supported():
        torch.cuda.manual_seed_all(seed)
