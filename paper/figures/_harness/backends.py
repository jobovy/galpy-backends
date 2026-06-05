"""Backend discovery + device labelling for comparison/benchmark figures."""


def available_backends():
    """Backends importable in the current environment, numpy always first."""
    backs = ["numpy"]
    try:
        import jax  # noqa: F401

        backs.append("jax")
    except Exception:
        pass
    try:
        import torch  # noqa: F401

        backs.append("torch")
    except Exception:
        pass
    return backs


def device_label(backend):
    """Short device string ('cpu'/'gpu'/'cuda') for a backend, for captions."""
    if backend == "jax":
        try:
            import jax

            return jax.devices()[0].platform  # 'cpu' or 'gpu'
        except Exception:
            return "?"
    if backend == "torch":
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "?"
    return "cpu"


def has_gpu():
    """True if any installed backend reports a GPU/CUDA device."""
    try:
        import jax

        if any(d.platform == "gpu" for d in jax.devices()):
            return True
    except Exception:
        pass
    try:
        import torch

        if torch.cuda.is_available():
            return True
    except Exception:
        pass
    return False
