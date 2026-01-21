from multiprocessing import cpu_count
import torch


def find_n_proc():
    return max(cpu_count() - 1, 1)

def find_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")