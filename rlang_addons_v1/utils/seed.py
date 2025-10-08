import os, random
try:
    import numpy as np
except Exception:
    np = None

def set_seeds(seed: int = 1337, deterministic: bool = True):
    """Set strong reproducibility seeds. Torch is optional.
    If torch is unavailable, silently skip its flags.
    """
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    if np is not None:
        np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        # Prefer stricter matmul precision (PyTorch 2.0+)
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    except Exception:
        # Torch not installed; that's fine for non-GPU pilot runs.
        pass
