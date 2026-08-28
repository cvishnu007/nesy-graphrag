import os


def available_cpu_workers(env_name, reserve=1):
    """Return an overrideable worker count while leaving room for the OS."""
    cpu_count = os.cpu_count() or 2
    default = max(1, cpu_count - max(0, reserve))
    return max(1, min(cpu_count, int(os.getenv(env_name, default))))


def configure_torch():
    """Configure PyTorch resources and return the best available device."""
    import torch

    thread_count = available_cpu_workers("TORCH_NUM_THREADS")
    interop_count = max(
        1,
        min(thread_count, int(os.getenv("TORCH_INTEROP_THREADS", "4"))),
    )
    torch.set_num_threads(thread_count)
    try:
        torch.set_num_interop_threads(interop_count)
    except RuntimeError:
        # PyTorch only permits changing this before parallel work begins.
        interop_count = torch.get_num_interop_threads()

    requested = os.getenv("EMBEDDING_DEVICE", "auto").strip().lower()
    if requested not in {"auto", "cuda", "mps", "cpu"}:
        raise ValueError("EMBEDDING_DEVICE must be auto, cuda, mps, or cpu")

    cuda_available = torch.cuda.is_available()
    mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    if requested == "cuda" and not cuda_available:
        raise RuntimeError("CUDA was requested but this PyTorch build cannot access it")
    if requested == "mps" and not mps_available:
        raise RuntimeError("MPS was requested but it is unavailable")

    if requested == "cuda" or (requested == "auto" and cuda_available):
        device = "cuda"
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        gpu = torch.cuda.get_device_properties(0)
        memory_gb = gpu.total_memory / (1024 ** 3)
        detail = f"{gpu.name}, {memory_gb:.1f} GB"
    elif requested == "mps" or (requested == "auto" and mps_available):
        device = "mps"
        detail = "Apple Metal"
    else:
        device = "cpu"
        detail = f"{thread_count} intra-op, {interop_count} inter-op threads"

    print(f"Embedding device: {device} ({detail})")
    return device
