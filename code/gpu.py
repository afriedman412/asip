import torch


def has_gpu():
    """
    Returns True if an NVIDIA GPU is present *and*
    can be used by cuML (RAPIDS).
    """
    try:
        import torch
        if torch.cuda.is_available():
            print("GPU Found")
            return True
    except Exception:
        pass

    # backup check via nvidia-smi
    try:
        import subprocess
        out = subprocess.run(
            ["nvidia-smi"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        return out.returncode == 0
    except Exception:
        return False


# ---- conditional imports ----

GPU_AVAILABLE = has_gpu()

if GPU_AVAILABLE:
    try:
        # RAPIDS UMAP, PCA, etc.
        print("loading cuml...")
        from cuml.preprocessing import StandardScaler
        from cuml.decomposition import PCA
        USING_GPU = True
    except Exception:
        # cuML not installed → fallback
        print("GPU not found ... loading sklearn...")
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        USING_GPU = False
else:
    print("error ... loading sklearn...")
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    USING_GPU = False

# ---- export flags ----
__all__ = [
    "PCA",
    "UMAP",
    "GPU_AVAILABLE",
    "USING_GPU"
]
